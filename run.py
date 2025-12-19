from classes import GARCH_GBM_StochasticDrift_Return, InvestmentSimulator, inflation_adjusted_value, annual_to_monthly_ror
strategy = GARCH_GBM_StochasticDrift_Return()

sim = InvestmentSimulator(
    start_money=0,
    monthly_investment=[200, 200, 0, 0, 0, 0, 0, 0],
    years=40,
    strategy=strategy,
    n_simulations=20000,
    annual_inflation=0.026
)

sim.run()

sim.export_json_percentiles()
sim.export_json_histogram()
sim.export_json_sample_paths(n_paths=20)
sim.export_json_contributions()
