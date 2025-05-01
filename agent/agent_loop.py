from agent.data_fetch import fetch_intraday_data
from agent.strategy import generate_strategy, execute_strategy, improve_strategy
from agent.eval import backtest_strategy, plot_backtest


def run_agent(df, ticker="TSLA", max_cycles=6, target_return=7.0):
    code, description = generate_strategy(df)
    
    while cycle <= max_cycles:
        print(f"\n🌀 [CYCLE {cycle}] Running strategy iteration...")

        # Step 1: Execute code
        try:
            df_cycle = execute_strategy(df.copy(), code)
        except Exception as e:
            print(f"[ERROR] Failed to execute strategy in cycle {cycle}: {e}")
            break

        # Step 2: Backtest
        results_str, results_dict, df_cycle = backtest_strategy(
            df_cycle, capital=10000, fee_per_trade=0.001, verbose=False
        )

        print(results_str)

        # Step 3: Check stopping condition
        total_return = results_dict.get("Total Return (%)", 0.0)
        if total_return >= target_return:
            print(f"\n✅ Target achieved! Strategy returned {total_return:.2f}%.")
            return code, description, results_str, df_cycle

        # Step 4: Try improving the strategy
        try:
            improved_code, improved_description = improve_strategy(df_cycle, code, results_str, ticker=ticker)
        except Exception as e:
            print(f"[ERROR] Strategy improvement failed in cycle {cycle}: {e}")
            break

        # Prepare for next cycle
        code, description = improved_code, improved_description
        cycle += 1

    print(f"\n🛑 Reached max cycles without hitting target return ({target_return:.2f}%).")
    return code, description, results_str, df_cycle