import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

def load_all_datasets():
    """Load all three datasets"""
    datasets = {}
    
    # 1. Redemptions (points redeemed)
    try:
        redemptions_path = r"C:\Users\user\OneDrive - Singapore Management University\Desktop\smu\subject\y4s2\capstone\cleaned data\redemptions_all.xlsx"
        df_redemptions = pd.read_excel(redemptions_path)
        print(f"✅ Loaded redemptions: {len(df_redemptions)} records")
        datasets['redemptions'] = df_redemptions
    except Exception as e:
        print(f"❌ Error loading redemptions: {e}")
        return None
    
    # 2. Mall Member (points earned)
    try:
        # Find mall member file - adjust pattern as needed
        import glob
        member_files = glob.glob(r"C:\Users\user\OneDrive - Singapore Management University\Desktop\smu\subject\y4s2\capstone\cleaned data\*mall_member*.xlsx")
        if member_files:
            df_member = pd.read_excel(member_files[0])
            print(f"✅ Loaded member data: {len(df_member)} records")
            datasets['member'] = df_member
        else:
            print("⚠️ No mall member file found")
            datasets['member'] = None
    except Exception as e:
        print(f"❌ Error loading member data: {e}")
        datasets['member'] = None
    
    # 3. GTO data (revenue)
    try:
        # Load GTO monthly sales for revenue
        gto_path = r"C:\Users\user\OneDrive - Singapore Management University\Desktop\smu\subject\y4s2\capstone\cleaned data\gto_monthly_sales*.xlsx"
        gto_files = glob.glob(gto_path)
        if gto_files:
            df_gto = pd.read_excel(gto_files[0])
            print(f"✅ Loaded GTO data: {len(df_gto)} records")
            datasets['gto'] = df_gto
        else:
            print("⚠️ No GTO file found")
            datasets['gto'] = None
    except Exception as e:
        print(f"❌ Error loading GTO data: {e}")
        datasets['gto'] = None
    
    return datasets

def analyze_integrated_kpis(datasets):
    """
    Integrated analysis of all three KPIs:
    1. Revenue per campaign (linking redemptions → GTO via outlet + date)
    2. Revenue uplift during campaign periods
    3. Revenue per point issued (member points → GTO revenue)
    """
    
    print("="*80)
    print("INTEGRATED KPI ANALYSIS: REDEMPTIONS → MEMBER → GTO")
    print("="*80)
    
    df_redemptions = datasets['redemptions']
    df_member = datasets['member']
    df_gto = datasets['gto']
    
    # KPI 1: REVENUE PER CAMPAIGN (Redemptions + GTO)
    print("\n" + "="*80)
    print("KPI 1: REVENUE PER CAMPAIGN")
    print("Linking redemptions (campaigns) to GTO revenue via outlet + date")
    print("="*80)
    
    if df_gto is not None and df_redemptions is not None:
        # Prepare data for linking
        # Extract month from transaction_date in redemptions
        df_redemptions['redemption_month'] = pd.to_datetime(df_redemptions['transaction_date']).dt.to_period('M')
        
        # Extract month from GTO reporting month
        # Assuming GTO has a month column - adjust based on your actual GTO data
        if 'gto_reporting_month' in df_gto.columns:
            df_gto['gto_month'] = pd.to_datetime(df_gto['gto_reporting_month']).dt.to_period('M')
        else:
            # Try to find a date column in GTO
            date_cols = [col for col in df_gto.columns if 'date' in col.lower() or 'month' in col.lower()]
            if date_cols:
                df_gto['gto_month'] = pd.to_datetime(df_gto[date_cols[0]]).dt.to_period('M')
            else:
                print("❌ No date column found in GTO data")
                df_gto['gto_month'] = None
        
        # Group redemptions by campaign, outlet, and month
        redemption_summary = df_redemptions.groupby(['voucher_code', 'outlet_code', 'redemption_month']).agg({
            'voucher_value': ['sum', 'count']
        }).reset_index()
        
        redemption_summary.columns = ['voucher_code', 'outlet_code', 'redemption_month', 
                                      'points_redeemed', 'redemption_count']
        
        # Group GTO by outlet and month
        if 'unit_no' in df_gto.columns:
            # Use unit_no as outlet identifier
            gto_summary = df_gto.groupby(['unit_no', 'gto_month']).agg({
                'total_GTO': 'sum'
            }).reset_index()
            gto_summary.columns = ['outlet_code', 'gto_month', 'gto_revenue']
        else:
            print("❌ No outlet identifier found in GTO data")
            gto_summary = None
        
        # Link redemptions to GTO revenue
        if gto_summary is not None:
            # Merge on outlet_code and month
            campaign_revenue = pd.merge(
                redemption_summary,
                gto_summary,
                left_on=['outlet_code', 'redemption_month'],
                right_on=['outlet_code', 'gto_month'],
                how='left'
            )
            
            # Calculate revenue per point
            campaign_revenue['revenue_per_point'] = campaign_revenue['gto_revenue'] / campaign_revenue['points_redeemed']
            
            # Summarize by campaign
            campaign_summary = campaign_revenue.groupby('voucher_code').agg({
                'gto_revenue': 'sum',
                'points_redeemed': 'sum',
                'redemption_count': 'sum'
            }).reset_index()
            
            campaign_summary['revenue_per_point'] = campaign_summary['gto_revenue'] / campaign_summary['points_redeemed']
            
            print(f"\n📊 Revenue per Campaign Analysis:")
            print(f"   Linked {len(campaign_revenue)} redemption periods to GTO revenue")
            
            # Sort by total revenue
            top_campaigns = campaign_summary.sort_values('gto_revenue', ascending=False).head(10)
            print(f"\n🏆 TOP 10 CAMPAIGNS BY REVENUE:")
            for idx, row in top_campaigns.iterrows():
                print(f"   {row['voucher_code']}: ${row['gto_revenue']:,.2f} revenue, {row['points_redeemed']:,.0f} points")
            
            # Save results
            campaign_summary.to_excel('campaign_revenue_linked_to_gto.xlsx', index=False)
            print("✅ Saved to 'campaign_revenue_linked_to_gto.xlsx'")
            
            # Regression: What drives campaign revenue?
            if len(campaign_summary) > 5:
                print("\n📊 REGRESSION: Campaign Revenue Drivers")
                
                # Prepare features
                X = campaign_summary[['points_redeemed', 'redemption_count']]
                y = campaign_summary['gto_revenue']
                
                model = LinearRegression()
                model.fit(X, y)
                
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                
                print(f"   Model: Revenue = {model.intercept_:.2f} + {model.coef_[0]:.3f}×Points + {model.coef_[1]:.2f}×Redemptions")
                print(f"   R²: {r2:.4f}")
                print(f"   Each point redeemed predicts ${model.coef_[0]:.3f} in revenue")
                print(f"   Each redemption predicts ${model.coef_[1]:.2f} in revenue")
    
    # KPI 2: REVENUE UPLIFT DURING CAMPAIGN PERIODS
    print("\n" + "="*80)
    print("KPI 2: REVENUE UPLIFT DURING CAMPAIGN PERIODS")
    print("Comparing GTO revenue during vs before campaign periods")
    print("="*80)
    
    if df_gto is not None and df_redemptions is not None and 'gto_month' in df_gto.columns:
        # Identify campaign periods from redemptions
        campaign_periods = df_redemptions.groupby('voucher_code').agg({
            'transaction_date': ['min', 'max']
        }).reset_index()
        campaign_periods.columns = ['voucher_code', 'campaign_start', 'campaign_end']
        
        # Convert to datetime and get month periods
        campaign_periods['campaign_start'] = pd.to_datetime(campaign_periods['campaign_start'])
        campaign_periods['campaign_end'] = pd.to_datetime(campaign_periods['campaign_end'])
        campaign_periods['campaign_start_month'] = campaign_periods['campaign_start'].dt.to_period('M')
        campaign_periods['campaign_end_month'] = campaign_periods['campaign_end'].dt.to_period('M')
        
        # For each outlet with GTO data, calculate uplift
        uplift_results = []
        
        # Get unique outlets from GTO
        if 'unit_no' in df_gto.columns:
            outlets = df_gto['unit_no'].unique()[:10]  # Limit to 10 for demonstration
        else:
            outlets = []
        
        for outlet in outlets:
            outlet_gto = df_gto[df_gto['unit_no'] == outlet]
            
            # For each campaign that occurred at this outlet
            outlet_redemptions = df_redemptions[df_redemptions['outlet_code'] == outlet]
            outlet_campaigns = outlet_redemptions['voucher_code'].unique()
            
            for campaign in outlet_campaigns[:3]:  # Limit to 3 campaigns per outlet
                campaign_dates = campaign_periods[campaign_periods['voucher_code'] == campaign]
                if len(campaign_dates) > 0:
                    campaign_start = campaign_dates['campaign_start'].iloc[0]
                    campaign_end = campaign_dates['campaign_end'].iloc[0]
                    
                    # Calculate revenue during campaign
                    campaign_revenue = outlet_gto[
                        (pd.to_datetime(outlet_gto['gto_reporting_month']) >= campaign_start) &
                        (pd.to_datetime(outlet_gto['gto_reporting_month']) <= campaign_end)
                    ]['total_GTO'].sum()
                    
                    # Calculate revenue during control period (same duration before campaign)
                    control_start = campaign_start - (campaign_end - campaign_start) - timedelta(days=1)
                    control_end = campaign_start - timedelta(days=1)
                    
                    control_revenue = outlet_gto[
                        (pd.to_datetime(outlet_gto['gto_reporting_month']) >= control_start) &
                        (pd.to_datetime(outlet_gto['gto_reporting_month']) <= control_end)
                    ]['total_GTO'].sum()
                    
                    if control_revenue > 0:
                        uplift_pct = ((campaign_revenue - control_revenue) / control_revenue) * 100
                        
                        uplift_results.append({
                            'outlet': outlet,
                            'campaign': campaign,
                            'campaign_revenue': campaign_revenue,
                            'control_revenue': control_revenue,
                            'uplift_pct': uplift_pct,
                            'uplift_abs': campaign_revenue - control_revenue
                        })
        
        if uplift_results:
            uplift_df = pd.DataFrame(uplift_results)
            
            print(f"\n📊 Revenue Uplift Analysis:")
            print(f"   Analyzed {len(uplift_df)} campaign-outlet combinations")
            print(f"   Average uplift: {uplift_df['uplift_pct'].mean():.1f}%")
            print(f"   Max uplift: {uplift_df['uplift_pct'].max():.1f}%")
            
            # Save results
            uplift_df.to_excel('revenue_uplift_analysis.xlsx', index=False)
            print("✅ Saved to 'revenue_uplift_analysis.xlsx'")
    
    # KPI 3: REVENUE PER POINT ISSUED (Member → GTO)
    print("\n" + "="*80)
    print("KPI 3: REVENUE PER POINT ISSUED")
    print("Linking member points earned to GTO revenue")
    print("="*80)
    
    if df_member is not None and df_gto is not None:
        # Prepare member data
        df_member['transaction_month'] = pd.to_datetime(df_member['transaction_date']).dt.to_period('M')
        
        # Aggregate points earned by outlet and month
        points_by_outlet_month = df_member.groupby(['outlet_code', 'transaction_month']).agg({
            'points_earned': 'sum',
            'amount': 'sum'  # Customer spending
        }).reset_index()
        
        # Prepare GTO data
        if 'gto_reporting_month' in df_gto.columns:
            df_gto['gto_month'] = pd.to_datetime(df_gto['gto_reporting_month']).dt.to_period('M')
        
        # Aggregate GTO revenue by outlet and month
        # Need to map GTO outlet to member outlet (both use outlet_code)
        gto_by_outlet_month = df_gto.groupby(['unit_no', 'gto_month']).agg({
            'total_GTO': 'sum'
        }).reset_index()
        gto_by_outlet_month.columns = ['outlet_code', 'gto_month', 'gto_revenue']
        
        # Merge points earned with GTO revenue
        # Convert periods to string for merging
        points_by_outlet_month['month_str'] = points_by_outlet_month['transaction_month'].astype(str)
        gto_by_outlet_month['month_str'] = gto_by_outlet_month['gto_month'].astype(str)
        
        merged_data = pd.merge(
            points_by_outlet_month,
            gto_by_outlet_month,
            left_on=['outlet_code', 'month_str'],
            right_on=['outlet_code', 'month_str'],
            how='inner'
        )
        
        # Calculate revenue per point
        merged_data['revenue_per_point'] = merged_data['gto_revenue'] / merged_data['points_earned']
        merged_data['revenue_per_dollar_spent'] = merged_data['gto_revenue'] / merged_data['amount']
        
        print(f"\n📊 Revenue per Point Analysis:")
        print(f"   Successfully linked {len(merged_data)} outlet-months")
        print(f"   Average revenue per point: ${merged_data['revenue_per_point'].mean():.4f}")
        print(f"   Average revenue per dollar spent: ${merged_data['revenue_per_dollar_spent'].mean():.2f}")
        
        # Save detailed results
        merged_data.to_excel('revenue_per_point_detailed.xlsx', index=False)
        
        # Summary statistics
        summary_stats = merged_data.agg({
            'points_earned': ['sum', 'mean'],
            'gto_revenue': ['sum', 'mean'],
            'revenue_per_point': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        summary_stats.to_excel('revenue_per_point_summary.xlsx')
        
        print("\n📊 Summary Statistics:")
        print(f"   Total points earned: {merged_data['points_earned'].sum():,.0f}")
        print(f"   Total GTO revenue: ${merged_data['gto_revenue'].sum():,.2f}")
        print(f"   Overall revenue per point: ${merged_data['gto_revenue'].sum() / merged_data['points_earned'].sum():.4f}")
        
        # Regression: Predict GTO revenue from points earned
        print("\n📊 REGRESSION: Points Earned → GTO Revenue")
        
        X = merged_data[['points_earned', 'amount']]
        y = merged_data['gto_revenue']
        
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        print(f"   Model: GTO Revenue = {model.intercept_:.2f} + {model.coef_[0]:.3f}×Points + {model.coef_[1]:.2f}×Spending")
        print(f"   R²: {r2:.4f} (explains {r2*100:.1f}% of variance)")
        print(f"   Each point earned predicts ${model.coef_[0]:.3f} in GTO revenue")
        print(f"   Each dollar spent predicts ${model.coef_[1]:.2f} in GTO revenue")
        
        print("✅ Saved detailed results to 'revenue_per_point_detailed.xlsx'")
    
    # Generate final report
    print("\n" + "="*80)
    print("FINAL INTEGRATED REPORT")
    print("="*80)
    
    print("\n🎯 KEY INSIGHTS:")
    
    # Insight 1: Best performing campaigns
    if 'campaign_summary' in locals():
        best_campaign = campaign_summary.loc[campaign_summary['gto_revenue'].idxmax()]
        print(f"1. BEST CAMPAIGN: {best_campaign['voucher_code']}")
        print(f"   • Generated ${best_campaign['gto_revenue']:,.2f} in revenue")
        print(f"   • ${best_campaign['revenue_per_point']:.4f} revenue per point redeemed")
    
    # Insight 2: Revenue uplift
    if 'uplift_df' in locals():
        avg_uplift = uplift_df['uplift_pct'].mean()
        print(f"2. CAMPAIGN EFFECTIVENESS: Average revenue uplift of {avg_uplift:.1f}%")
        print(f"   • {len(uplift_df[uplift_df['uplift_pct'] > 0])} campaigns had positive uplift")
        print(f"   • Best uplift: {uplift_df['uplift_pct'].max():.1f}%")
    
    # Insight 3: Points efficiency
    if 'merged_data' in locals():
        avg_rev_per_point = merged_data['revenue_per_point'].mean()
        print(f"3. LOYALTY PROGRAM EFFICIENCY: ${avg_rev_per_point:.4f} revenue per point issued")
        print(f"   • For every 100 points issued, expect ${avg_rev_per_point * 100:.2f} in revenue")
        
        # Find most efficient outlets
        efficient_outlets = merged_data.groupby('outlet_code')['revenue_per_point'].mean().nlargest(3)
        print(f"   • Most efficient outlets:")
        for outlet, value in efficient_outlets.items():
            print(f"     - {outlet}: ${value:.4f} revenue per point")
    
    print("\n💡 RECOMMENDATIONS:")
    print("1. Focus campaigns on high-GTO outlets")
    print("2. Time campaigns to match high-revenue months")
    print("3. Monitor revenue-per-point by outlet to optimize loyalty spending")
    
    print("="*80)
    
    # Create integrated visualization
    create_integrated_visualizations(locals())

def create_integrated_visualizations(local_vars):
    """Create comprehensive visualizations"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. Campaign revenue distribution
    if 'campaign_summary' in local_vars:
        top_10 = local_vars['campaign_summary'].nlargest(10, 'gto_revenue')
        axes[0, 0].barh(top_10['voucher_code'], top_10['gto_revenue'])
        axes[0, 0].set_xlabel('GTO Revenue ($)')
        axes[0, 0].set_title('Top 10 Campaigns by Revenue')
        axes[0, 0].invert_yaxis()
    
    # 2. Revenue per point by campaign
    if 'campaign_summary' in local_vars:
        efficient_campaigns = local_vars['campaign_summary'].nlargest(10, 'revenue_per_point')
        axes[0, 1].barh(efficient_campaigns['voucher_code'], efficient_campaigns['revenue_per_point'])
        axes[0, 1].set_xlabel('Revenue per Point ($)')
        axes[0, 1].set_title('Most Efficient Campaigns')
        axes[0, 1].invert_yaxis()
    
    # 3. Revenue uplift distribution
    if 'uplift_df' in local_vars:
        axes[1, 0].hist(local_vars['uplift_df']['uplift_pct'], bins=20, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(local_vars['uplift_df']['uplift_pct'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {local_vars["uplift_df"]["uplift_pct"].mean():.1f}%')
        axes[1, 0].set_xlabel('Revenue Uplift (%)')
        axes[1, 0].set_ylabel('Number of Campaigns')
        axes[1, 0].set_title('Distribution of Campaign Uplift')
        axes[1, 0].legend()
    
    # 4. Points earned vs GTO revenue
    if 'merged_data' in local_vars:
        sample = local_vars['merged_data'].sample(min(50, len(local_vars['merged_data'])), random_state=42)
        axes[1, 1].scatter(sample['points_earned'], sample['gto_revenue'], alpha=0.6)
        axes[1, 1].set_xlabel('Points Earned')
        axes[1, 1].set_ylabel('GTO Revenue ($)')
        axes[1, 1].set_title('Points Earned vs GTO Revenue')
        
        # Add trend line
        z = np.polyfit(sample['points_earned'], sample['gto_revenue'], 1)
        p = np.poly1d(z)
        axes[1, 1].plot(sample['points_earned'], p(sample['points_earned']), "r--", alpha=0.8)
    
    # 5. Revenue per point by outlet
    if 'merged_data' in local_vars:
        outlet_efficiency = local_vars['merged_data'].groupby('outlet_code')['revenue_per_point'].mean().nlargest(10)
        axes[2, 0].barh(outlet_efficiency.index, outlet_efficiency.values)
        axes[2, 0].set_xlabel('Revenue per Point ($)')
        axes[2, 0].set_title('Top 10 Outlets by Revenue Efficiency')
        axes[2, 0].invert_yaxis()
    
    # 6. Summary text
    summary_text = "INTEGRATED ANALYSIS\n"
    if 'campaign_summary' in local_vars:
        total_campaign_revenue = local_vars['campaign_summary']['gto_revenue'].sum()
        summary_text += f"Campaign Revenue: ${total_campaign_revenue:,.0f}\n"
    
    if 'uplift_df' in local_vars:
        avg_uplift = local_vars['uplift_df']['uplift_pct'].mean()
        summary_text += f"Avg Uplift: {avg_uplift:.1f}%\n"
    
    if 'merged_data' in local_vars:
        overall_rev_per_point = local_vars['merged_data']['gto_revenue'].sum() / local_vars['merged_data']['points_earned'].sum()
        summary_text += f"Rev/Point: ${overall_rev_per_point:.4f}"
    
    axes[2, 1].text(0.5, 0.5, summary_text, 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=axes[2, 1].transAxes,
                   fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[2, 1].axis('off')
    axes[2, 1].set_title('Key Metrics Summary')
    
    plt.suptitle('Integrated KPI Analysis: Redemptions → Member → GTO', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('integrated_kpi_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution"""
    print("="*80)
    print("STARTING INTEGRATED KPI ANALYSIS")
    print("="*80)
    
    # Load all datasets
    datasets = load_all_datasets()
    
    if datasets:
        # Run integrated analysis
        analyze_integrated_kpis(datasets)
    else:
        print("❌ Failed to load necessary datasets")

if __name__ == "__main__":
    main()