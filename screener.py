"""Screening saham Indonesia + brief pagi/sore.
Pakai: python screener.py morning   |   python screener.py evening
Daftar saham dibaca dari tickers.txt (satu kode per baris, tanpa .JK).
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

from monitor import ask_ai, fetch_crypto, fmt_news, google_news, send

# ====================== PENGATURAN ======================
MIN_AVG_VALUE = 5e9    # filter likuiditas: rata-rata nilai transaksi 20 hari >= Rp 5 miliar
MIN_ROWS = 120         # minimal hari data
TOP_N = 5              # jumlah kandidat
CHUNK = 100            # jumlah saham per unduhan
# ========================================================


def load_tickers():
    with open("tickers.txt") as f:
        return sorted({l.strip().upper().replace(".JK", "") for l in f if l.strip()})


def download(tickers):
    out = {}
    for i in range(0, len(tickers), CHUNK):
        chunk = [t + ".JK" for t in tickers[i:i + CHUNK]]
        try:
            df = yf.download(chunk, period="1y", interval="1d", group_by="ticker",
                             auto_adjust=True, progress=False, threads=True)
        except Exception as e:
            print("download error:", e)
            continue
        for t in chunk:
            try:
                d = df[t].dropna(subset=["Close"]) if isinstance(df.columns, pd.MultiIndex) else df.dropna(subset=["Close"])
                if len(d):
                    out[t[:-3]] = d
            except Exception:
                pass
        time.sleep(1)
    return out


def rsi14(close):
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def analyze(d):
    """Hitung indikator + skor satu saham. Return None kalau tidak lolos filter."""
    if len(d) < MIN_ROWS:
        return None
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    price = float(c.iloc[-1])
    value20 = float((c * v).tail(20).mean())
    if value20 < MIN_AVG_VALUE or price <= 0:
        return None
    ma20, ma50 = c.tail(20).mean(), c.tail(50).mean()
    ma200 = c.tail(200).mean() if len(c) >= 200 else None
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    vol_ratio = float(v.iloc[-1] / v.tail(20).mean()) if v.tail(20).mean() > 0 else 0
    ret = lambda n: float(c.iloc[-1] / c.iloc[-1 - n] - 1) if len(c) > n else 0.0
    r1, r5, r20, r60 = ret(1), ret(5), ret(20), ret(60)
    rsi = rsi14(c)
    dist_high = float(price / c.tail(60).max() - 1)
    low20 = float(l.tail(20).min())

    score = 0.0
    if price > ma20 > ma50:
        score += 20
    if ma200 is not None and price > ma200:
        score += 10
    score += min(20, max(0.0, r20 * 100))
    if 1.5 <= vol_ratio <= 5:
        score += 15
    elif vol_ratio >= 1.0:
        score += 7
    if dist_high >= -0.03:
        score += 15
    elif dist_high >= -0.08:
        score += 8
    if 50 <= rsi <= 68:
        score += 10
    if rsi > 75:
        score -= 15
    if r5 > 0.20:
        score -= 10
    if r60 < -0.20:
        score -= 10

    risk_level = max(low20, price - 1.5 * atr)
    return {
        "kode": "", "harga": price, "skor": round(score, 1), "r1d": r1 * 100, "r5d": r5 * 100,
        "r20d": r20 * 100, "r60d": r60 * 100, "rsi": rsi, "vol_x": vol_ratio,
        "dari_high60": dist_high * 100, "nilai20": value20,
        "acuan_risiko": risk_level, "risiko_pct": (price - risk_level) / price * 100,
    }


def ihsg_line():
    try:
        d = yf.Ticker("^JKSE").history(period="5d", interval="1d")["Close"].dropna()
        return f"IHSG {d.iloc[-1]:,.0f} ({(d.iloc[-1] / d.iloc[-2] - 1) * 100:+.2f}% hari terakhir)"
    except Exception:
        return "IHSG: data tidak tersedia"


def table(rows, cols):
    return "\n".join(
        f"{r['kode']}: " + ", ".join(f"{k} {r[k]:,.1f}" for k in cols) for r in rows)


def main(mode):
    tickers = load_tickers()
    data = download(tickers)
    results = []
    for code, d in data.items():
        try:
            r = analyze(d)
            if r:
                r["kode"] = code
                results.append(r)
        except Exception as e:
            print(code, "error:", e)
    if not results:
        send("Screening gagal: tidak ada data saham yang berhasil diambil.")
        return

    scr = sorted(results, key=lambda x: x["skor"], reverse=True)
    picks = scr[:TOP_N]
    gainers = sorted(results, key=lambda x: x["r1d"], reverse=True)[:5]
    losers = sorted(results, key=lambda x: x["r1d"])[:5]
    volume = sorted(results, key=lambda x: x["vol_x"], reverse=True)[:5]

    news = {}
    for p in picks:
        try:
            news[p["kode"]] = google_news(f"saham {p['kode']}", 3, 72 * 60)
        except Exception:
            news[p["kode"]] = []

    try:
        cr = fetch_crypto()
        crypto_line = ", ".join(f"{k} ${v['price']:,.0f} ({v['day']:+.1f}%/24j)" for k, v in cr.items())
    except Exception:
        crypto_line = "tidak tersedia"

    cols = ["harga", "skor", "r1d", "r20d", "rsi", "vol_x", "dari_high60", "acuan_risiko", "risiko_pct"]
    judul = "BRIEF PAGI (sebelum market buka)" if mode == "morning" else "BRIEF SORE (market tutup)"
    fokus = ("Fokus: rencana & hal yang dipantau hari ini, berdasarkan data penutupan terakhir."
             if mode == "morning" else
             "Fokus: rekap hari ini (yang naik/turun/ramai) dan kandidat untuk dipantau besok.")

    prompt = (
        f"{judul}. {fokus}\n\n{ihsg_line()}\nCrypto: {crypto_line}\n"
        f"Jumlah saham lolos filter likuiditas: {len(results)} dari {len(tickers)}\n\n"
        f"KANDIDAT TERATAS (skor teknikal 0-100; r1d/r20d = % return; vol_x = volume vs rata-rata 20 hari; "
        f"dari_high60 = jarak ke harga tertinggi 60 hari; acuan_risiko = level di bawah harga yang kalau jebol "
        f"berarti skenario naik melemah):\n{table(picks, cols)}\n\n"
        "BERITA KANDIDAT:\n" + "\n".join(f"[{k}]\n{fmt_news(v)}" for k, v in news.items()) +
        f"\n\nTOP NAIK: {table(gainers, ['harga', 'r1d'])}\nTOP TURUN: {table(losers, ['harga', 'r1d'])}\n"
        f"VOLUME TERAMAI: {table(volume, ['harga', 'vol_x', 'r1d'])}\n\n"
        "Susun brief dengan format: 1) RINGKASAN PASAR (IHSG + crypto, 2-3 kalimat). "
        "2) KANDIDAT SAHAM (tiap saham: alasan teknikal dari data, katalis berita jika ada, "
        "level acuan risiko, skenario naik dan turun, tingkat keyakinan rendah/sedang/tinggi). "
        "3) PERGERAKAN MENARIK (naik/turun/volume) secukupnya. 4) RISIKO. "
        "Maks 400 kata. Katakan jelas bahwa kandidat adalah hasil screening teknikal, bukan "
        "ajakan beli, dan tidak ada jaminan hasil."
    )
    ai = ask_ai(prompt)
    raw = f"\n\nDATA SCREENING (top 10 skor):\n{table(scr[:10], ['harga', 'skor', 'r20d', 'rsi', 'vol_x'])}"
    head = f"{judul}\n\n"
    send(head + (ai if ai else f"{ihsg_line()}\n\nKANDIDAT:\n{table(picks, cols)}") + raw)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evening")
