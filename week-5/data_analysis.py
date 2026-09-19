import pandas as pd

df = pd.read_csv('cosmetics_sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M').astype(str)
df['Price_per_box'] = df['Amount ($)'] / df['Boxes Shipped']


sp = df.groupby('Sales Person').agg(
    Total_Revenue=('Amount ($)', 'sum'),
    Total_Boxes=('Boxes Shipped', 'sum'),
    Orders=('Amount ($)', 'count')
).sort_values('Total_Revenue', ascending=False)
sp['Avg_Order_Value'] = sp['Total_Revenue'] / sp['Orders']

print("=== 1. SALES PERSON PERFORMANCE ===")
print(sp.round(2))
print()


monthly = df.groupby('Month')['Amount ($)'].sum().sort_index()
growth = monthly.pct_change() * 100

print("=== 2. MONTHLY REVENUE & GROWTH % ===")
result = pd.DataFrame({'Revenue': monthly.round(2), 'Growth_%': growth.round(2)})
print(result)
print(f"Strongest growth month: {growth.idxmax()} ({growth.max():.2f}%)")
print(f"Weakest/decline month: {growth.idxmin()} ({growth.min():.2f}%)")
print()


prod = df.groupby('Product').agg(
    Avg_Price_per_box=('Price_per_box', 'mean'),
    Total_Boxes=('Boxes Shipped', 'sum'),
    Total_Revenue=('Amount ($)', 'sum')
).sort_values('Avg_Price_per_box', ascending=False)

correlation = prod['Avg_Price_per_box'].corr(prod['Total_Boxes'])

print("=== 3. PRICE PER BOX vs VOLUME (by product) ===")
print(prod.round(2))
print(f"Correlation (avg price vs total boxes sold): {correlation:.3f}")
print()


country_total = df.groupby('Country')['Amount ($)'].sum()
country_product = df.groupby(['Country', 'Product'])['Amount ($)'].sum().reset_index()

top_product_per_country = country_product.loc[country_product.groupby('Country')['Amount ($)'].idxmax()]
top_product_per_country = top_product_per_country.merge(
    country_total.rename('Country_Total'), on='Country'
)
top_product_per_country['Concentration_%'] = (
    top_product_per_country['Amount ($)'] / top_product_per_country['Country_Total'] * 100
).round(2)
top_product_per_country = top_product_per_country.sort_values('Concentration_%', ascending=False)

print("=== 4. COUNTRY-PRODUCT CONCENTRATION RISK ===")
print(top_product_per_country[['Country', 'Product', 'Amount ($)', 'Country_Total', 'Concentration_%']].to_string(index=False))
print()
high_risk = top_product_per_country[top_product_per_country['Concentration_%'] > 40]
print("Countries with >40% concentration (high risk):")
print(high_risk[['Country', 'Product', 'Concentration_%']].to_string(index=False) if not high_risk.empty else "None — no country exceeds 40% concentration.")