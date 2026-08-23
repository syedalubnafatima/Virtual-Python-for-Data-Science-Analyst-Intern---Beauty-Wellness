import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cosmetics_sales_data.csv')

country_revenue = df.groupby('Country')['Amount ($)'].sum()
country_revenue = country_revenue.sort_values(ascending=False)
print(country_revenue)

plt.figure(figsize=(9,5))
plt.bar(country_revenue.index, country_revenue.values, color='indianred')
plt.title('Total Revenue by Country')
plt.xlabel('Country')
plt.ylabel('Total Amount ($)')
plt.savefig('charts/01_revenue_by_country.png', dpi=150, bbox_inches='tight')
plt.show()




# ----------------linepllot-----------------
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M').astype(str)

monthly_revenue = df.groupby('Month')['Amount ($)'].sum().sort_index()

plt.figure(figsize=(9,5))
plt.plot(monthly_revenue.index, monthly_revenue.values, marker='o', color='indianred', linewidth=2)
plt.title('Monthly Revenue Trend (Jan-Aug 2022)')
plt.xlabel('Month')
plt.ylabel('Total Amount ($)')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('charts/03_monthly_trend.png', dpi=150, bbox_inches='tight')
plt.show()

# --------------heat map--------------
pivot=df.pivot_table(index='Country', columns='Product', values='Amount ($)', aggfunc='sum', fill_value=0)

plt.figure(figsize=(12,5.5))
plt.imshow(pivot.values, cmap='OrRd', aspect='auto')
plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha='right')
plt.yticks(range(len(pivot.index)), pivot.index)
plt.title('Revenue Heatmap:COuntry vs. Product')
plt.colorbar(label='Total Revenue ($)')
plt.tight_layout()
plt.savefig('charts/06_heatmap_country_product.png', dpi=150, bbox_inches='tight')
plt.show()