import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ==============================================================================
# TRIAGE - LAYER 2: CONSTRAINED ALLOCATION OPTIMIZER
# ==============================================================================
# This script takes the Output of Layer 1 (The Severity Scores) and a Donor Budget 
# (e.g. $100M). It calculates the mathematically perfect way to distribute that 
# money across the active crises to maximize the TOTAL LIVES SAVED.
#
# CONSTRAINTS:
# 1. Diminishing Returns: The first $1M saves more lives than the $100th Million.
# 2. Access/Absorptive Capacity: A country cannot absorb infinite money if 
#    there is active conflict blocking aid trucks (OCHA Access Score).
# ==============================================================================

def diminishing_returns_curve(allocation, base_cost_per_life, access_penalty):
    """
    Models the non-linear impact of funding.
    Uses a logarithmic curve: Impact = log(allocation + 1) * scaling_factor
    """
    # If no allocation, no lives saved
    if allocation <= 0:
        return 0.0
        
    # The true cost per life increases if access is difficult (penalty)
    effective_cost = base_cost_per_life * (1 + access_penalty)
    
    # Mathematical heuristic for diminishing returns in humanitarian aid:
    # We use a square root function to simulate rapid early impact that tapers off.
    lives_saved = (np.sqrt(allocation) * 1000) / effective_cost
    return lives_saved

def objective_function(allocations, base_costs, access_penalties):
    """
    The function we want to MAXIMIZE (or in SciPy, minimize the negative).
    It calculates the total lives saved across all crises for a given allocation array.
    """
    total_lives_saved = 0
    for i in range(len(allocations)):
        lives = diminishing_returns_curve(allocations[i], base_costs[i], access_penalties[i])
        total_lives_saved += lives
        
    # We return the negative because SciPy's 'minimize' function looks for the lowest number.
    # Minimizing negative lives = Maximizing total lives.
    return -total_lives_saved


def run_allocation_optimizer(df_crises, total_budget_usd):
    """
    Runs the constrained optimization across the active crisis dataframe.
    """
    print(f"Running Optimization Engine for Budget: ${total_budget_usd:,.2f}")
    
    num_crises = len(df_crises)
    
    # Extract arrays for the math engine
    # NOTE: In a real scenario, you pull these from WHO/GiveWell historical data. 
    # For the MVP, we simulate base costs modifying them heavily by the Severity Score to force the AI to prioritize Red Zones.
    
    # We square the severity score effect to create exponential prioritization for the top crises
    base_severity_factor = (df_crises['Crisis_Severity_Score'] / 10) ** 2
    # Add a small epsilon to prevent division by zero
    base_costs = 50000 / (base_severity_factor + 0.1) 
    
    # Access Penalty (0.0 = perfect access, 1.0 = total war zone blocking aid) 
    # Simulated for the MVP based on conflict fatalities
    access_penalties = df_crises['fatalities'] / max(df_crises['fatalities'].max(), 1) 
    
    # ---------------------------------------------------------
    # CONSTRAINT 1: The total allocated money must equal the budget
    # ---------------------------------------------------------
    def budget_constraint(allocations):
        return total_budget_usd - np.sum(allocations)
    
    # ---------------------------------------------------------
    # CONSTRAINT 2: Bounds (Min $0, Max = Absorptive Capacity)
    # ---------------------------------------------------------
    # A country shouldn't receive more than its Total Unmet Need or the max budget
    bounds = []
    for index, row in df_crises.iterrows():
        unmet_need = max(0, row['funding_required'] - row['funding_received'])
        # Allow the algorithm to allocate up to 100% of the budget to one country if mathematically optimal
        max_cap = min(unmet_need, total_budget_usd)
        bounds.append((0, max_cap))
        
    # Initial Guess: Distribute money equally to start the algorithm
    initial_guess = np.full(num_crises, total_budget_usd / num_crises)
    
    # Run the SciPy SLSQP Optimizer
    result = minimize(
        objective_function,
        initial_guess,
        args=(base_costs, access_penalties),
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'eq', 'fun': budget_constraint}
    )
    
    # Append results to the dataframe
    df_crises['Optimal_Allocation_USD'] = np.round(result.x, 2)
    
    # Calculate exactly how many lives this specific allocation saves per country
    df_crises['Projected_Lives_Saved'] = [
        round(diminishing_returns_curve(alloc, cost, pen)) 
        for alloc, cost, pen in zip(result.x, base_costs, access_penalties)
    ]
    
    return df_crises

if __name__ == "__main__":
    import os
    
    # Check if we are running natively inside Databricks
    in_databricks = 'spark' in globals() or 'spark' in locals()
    
    if in_databricks:
        print("Loading Layer 1 scores from Databricks Unity Catalog...")
        df_test = spark.table("workspace.raw.triage_master_scores").toPandas()
    else:
        # Path to our Layer 1 output locally
        input_csv = "../assets/triage_master_scores.csv"
        
        if os.path.exists(input_csv):
            print(f"Loading data from {input_csv}...")
            df_test = pd.read_csv(input_csv)
        else:
            print(f"File not found: {input_csv}. Using mock data fallback.")
            mock_layer_1_output = {
                'iso3': ['SDN', 'UKR', 'HTI'], 
                'Crisis_Severity_Score': [92.5, 65.0, 88.0],
                'funding_required': [2700000000, 3100000000, 700000000],
                'funding_received': [405000000, 2500000000, 150000000],
                'fatalities': [800, 1200, 300]
            }
            df_test = pd.DataFrame(mock_layer_1_output)
    
    # Ensure fatalities column exists for the access penalty calculation
    if 'fatalities' not in df_test.columns:
        if 'fatalities_last_30_days' in df_test.columns:
            df_test.rename(columns={'fatalities_last_30_days': 'fatalities'}, inplace=True)
        else:
            df_test['fatalities'] = 0

    DONOR_BUDGET = 100_000_000 # $100 Million Dollar Simulation
    
    # Fill missing funding data with generic proxies so the optimizer doesn't crash empty
    # If the UN data is missing BOTH required and received, it means it's an un-tracked crisis (like a sudden outbreak),
    # so we assume a default $1B unmet need to allow the AI to allocate funds to it based on severity.
    df_test['funding_required'] = np.where((df_test['funding_required'] == 0) & (df_test['funding_received'] == 0), 1_000_000_000, df_test['funding_required'])
    
    df_test.fillna({'funding_required': 1_000_000_000, 'funding_received': 0}, inplace=True)

    # Only optimize countries with a severity score > 10 (Filtering out non-crises or perfectly funded places)
    df_active_crises = df_test[(df_test['Crisis_Severity_Score'] > 10) & (df_test['funding_required'] > df_test['funding_received'])].copy()
    
    if df_active_crises.empty:
      print("No active underfunded crises to optimize.")
      df_optimized = df_test
    else:
      df_active_crises.reset_index(drop=True, inplace=True)
      df_optimized = run_allocation_optimizer(df_active_crises, DONOR_BUDGET)
      
      print("\n================ OPTIMIZED PORTFOLIO DEPLOYMENT ================")
      print(df_optimized[['iso3', 'Crisis_Severity_Score', 'Optimal_Allocation_USD', 'Projected_Lives_Saved']].sort_values(by='Optimal_Allocation_USD', ascending=False).head(15))
      print(f"\nTOTAL LIVES SAVED GLOBALLY: {df_optimized['Projected_Lives_Saved'].sum():,}")
      
      try:
          if in_databricks:
              # Save the optimized file natively back to Databricks
              spark_df = spark.createDataFrame(df_optimized)
              spark_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.raw.triage_master_optimized")
              print("\nSaved optimized results to Unity Catalog: workspace.raw.triage_master_optimized")
          else:
              # Save locally for the Streamlit dashboard
              output_csv = "../assets/triage_master_optimized.csv"
              df_optimized.to_csv(output_csv, index=False)
              print(f"\nSaved optimized results to: {output_csv}")
      except Exception as e:
          print(f"Could not save output file: {e}")
