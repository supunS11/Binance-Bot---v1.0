from binance.client import Client

client = Client()

exchange_info = client.futures_exchange_info()

symbols = []

for s in exchange_info['symbols']:

    # ONLY USDT PERPETUALS
    if (
        s['contractType'] == 'PERPETUAL'
        and s['quoteAsset'] == 'USDT'
        and s['status'] == 'TRADING'
    ):

        symbols.append(s['symbol'])

# PRINT AS SINGLE LINE
print(",".join(symbols))

# TOTAL COUNT
print(f"\nTOTAL COINS: {len(symbols)}")