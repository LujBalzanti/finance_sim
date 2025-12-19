# Take an investment and calculate a projection of futures with each month having a chance of reducing or increasing return rate.

# I want to put in current state, when I started investing, how much was start investment, how much I put in each month. And then get a return rate from that calculation I could apply to "future".

# I think it'd be best if we split this into two different functionalitys.
import math
import random

class ReturnStrategy:
    """Base class for any return-application strategy."""
    def apply_return(self, balance: float) -> float:
        raise NotImplementedError("ReturnStrategy subclasses must override apply_return().")


class FixedMonthlyReturn(ReturnStrategy):
    """Applies a fixed monthly rate of return."""
    def __init__(self, monthly_ror: float):
        self.monthly_ror = monthly_ror

    def apply_return(self, balance: float) -> float:
        return balance * (1 + self.monthly_ror)
    

class IMPROVED_GARCH_GBM_Return(ReturnStrategy):
    """
    GBM returns with GARCH(1,1) volatility,
    Student-t shocks, and volatility cap.
    Calibrated for 30-year S&P 500 investing.
    """

    def __init__(self,
                 annual_drift=0.082,
                 annual_fee = 0.0007,
                 omega=0.000072,
                 alpha=0.09,
                 beta=0.88,
                 t_df=7,
                 vol_cap_annual=0.35):
        self.annual_fee = annual_fee
        self.monthly_fee = annual_fee / 12
        # Drift
        self.mu_annual = annual_drift
        self.mu_monthly = self.mu_annual / 12

        # GARCH parameters
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

        # Student-t degrees of freedom
        self.t_df = t_df

        # Volatility cap (monthly)
        self.vol_cap = vol_cap_annual / math.sqrt(12)

        # Initialize variance at unconditional level
        self.sigma2 = omega / (1 - alpha - beta)

        # Previous innovation εₜ₋₁
        self.prev_eps = 0.0

    def _student_t_shock(self):
        """
        Standardized Student-t shock (mean=0, var=1)
        """
        t = random.gauss(0, 1)
        chi2 = sum(random.gauss(0, 1) ** 2 for _ in range(self.t_df))
        return t / math.sqrt(chi2 / self.t_df)

    def apply_return(self, balance: float) -> float:

        # --- 1. Update conditional variance ---
        self.sigma2 = (
            self.omega +
            self.alpha * (self.prev_eps ** 2) +
            self.beta * self.sigma2
        )

        sigma = math.sqrt(self.sigma2)

        # --- 2. Volatility cap ---
        sigma = min(sigma, self.vol_cap)
        self.sigma2 = sigma ** 2

        # --- 3. Shock (Student-t) ---
        z = self._student_t_shock()
        eps = sigma * z

        # --- 4. Log-return with GBM correction ---
        r = self.mu_monthly - 0.5 * self.sigma2 + eps

        # Save innovation for next step
        self.prev_eps = eps

        # --- 5. Apply return ---
        return balance * math.exp(r - self.monthly_fee)

    def reset(self):
        self.sigma2 = self.omega / (1 - self.alpha - self.beta)
        self.prev_eps = 0.0

class GARCH_GBM_Return(ReturnStrategy):
    """
    GBM returns with volatility driven by GARCH(1,1).
    Produces:
    - Long-term drift (GBM)
    - Volatility clustering (GARCH)
    - Realistic long-term index-like paths
    """

    def __init__(self,
                 annual_drift=0.08,
                 omega=0.00002,
                 alpha=0.10,
                 beta=0.88):

        self.mu_annual = annual_drift
        self.mu_monthly = self.mu_annual / 12

        self.omega = omega
        self.alpha = alpha
        self.beta = beta

        self.sigma2 = omega / (1 - alpha - beta)

        self.prev_eps = 0.0

    def apply_return(self, balance: float) -> float:

        # 1. GARCH variance update
        self.sigma2 = (
            self.omega +
            self.alpha * (self.prev_eps ** 2) +
            self.beta * self.sigma2
        )
        sigma = math.sqrt(self.sigma2)

        # 2. Shock
        z = random.gauss(0, 1)
        eps = sigma * z

        # 3. Log-return (GBM-corrected)
        r = self.mu_monthly - 0.5 * self.sigma2 + eps

        self.prev_eps = eps

        return balance * math.exp(r)

    def reset(self):
        self.sigma2 = self.omega / (1 - self.alpha - self.beta)
        self.prev_eps = 0.0

class GARCH_GBM_StochasticDrift_Return:
    """
    GBM returns with GARCH(1,1) volatility, Student-t shocks, volatility cap,
    and mean-reverting stochastic drift.
    """

    def __init__(self,
                 long_run_drift=0.082,    # long-term annual mean drift
                 annual_fee=0.0007,
                 omega=0.000072,
                 alpha=0.09,
                 beta=0.88,
                 t_df=7,
                 vol_cap_annual=0.35,
                 drift_phi=0.95,          # mean reversion of drift
                 drift_sigma=0.01):       # volatility of drift shocks

        # --- Fees ---
        self.annual_fee = annual_fee
        self.monthly_fee = annual_fee / 12

        # --- GARCH parameters ---
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

        # --- Student-t shocks ---
        self.t_df = t_df

        # --- Volatility cap ---
        self.vol_cap = vol_cap_annual / math.sqrt(12)

        # --- Drift parameters ---
        self.mu_long = long_run_drift / 12      # monthly long-run mean
        self.phi = drift_phi                     # mean reversion
        self.sigma_eta = drift_sigma / math.sqrt(12)  # monthly drift shock
        self.mu_t = self.mu_long                 # initial drift

        # --- Initialize variance ---
        self.sigma2 = omega / (1 - alpha - beta)
        self.prev_eps = 0.0

    def _student_t_shock(self):
        """
        Standardized Student-t shock (mean=0, var=1)
        """
        t = random.gauss(0, 1)
        chi2 = sum(random.gauss(0, 1) ** 2 for _ in range(self.t_df))
        return t / math.sqrt(chi2 / self.t_df)

    def _update_drift(self):
        """
        AR(1) mean-reverting drift
        """
        shock = random.gauss(0, self.sigma_eta)
        self.mu_t = self.mu_long + self.phi * (self.mu_t - self.mu_long) + shock

    def apply_return(self, balance: float) -> float:

        # --- 1. Update drift ---
        self._update_drift()

        # --- 2. Update GARCH variance ---
        self.sigma2 = self.omega + self.alpha * (self.prev_eps ** 2) + self.beta * self.sigma2
        sigma = math.sqrt(self.sigma2)

        # --- 3. Volatility cap ---
        sigma = min(sigma, self.vol_cap)
        self.sigma2 = sigma ** 2

        # --- 4. Student-t shock ---
        z = self._student_t_shock()
        eps = sigma * z

        # --- 5. Log-return ---
        r = self.mu_t - 0.5 * self.sigma2 + eps

        # --- 6. Save innovation ---
        self.prev_eps = eps

        # --- 7. Apply return ---
        return balance * math.exp(r - self.monthly_fee)

    def reset(self):
        self.sigma2 = self.omega / (1 - self.alpha - self.beta)
        self.prev_eps = 0.0
        self.mu_t = self.mu_long
    

import numpy as np

import json

import random

class InvestmentSimulator:
    def __init__(self, start_money, monthly_investment, years, strategy,
                 n_simulations=1000, annual_inflation=0.02):

        self.start_money = start_money
        self.monthly_investment = monthly_investment
        self.years = years
        self.strategy = strategy
        self.n_simulations = n_simulations
        self.n_months = years * 12
        self.annual_inflation = annual_inflation

        self.simulations = None
        self.real_simulations = None

        # --- NEW: build month-by-month investment schedule ---
        self.investment_schedule = self._build_investment_schedule()

    def _build_investment_schedule(self):
        """
        Returns an array of length n_months defining the monthly investment.
        Supports:
        - a single number (constant contribution)
        - a list of tiered contributions applied over equal time segments
        """
        # Case 1: simple number
        if isinstance(self.monthly_investment, (int, float)):
            return np.full(self.n_months, float(self.monthly_investment))

        # Case 2: tiered list
        tiers = self.monthly_investment
        if not isinstance(tiers, (list, tuple)):
            raise ValueError("monthly_investment must be a number or list")

        n_tiers = len(tiers)
        months_per_tier = self.n_months // n_tiers

        schedule = []

        # add full tiers
        for amount in tiers:
            schedule.extend([amount] * months_per_tier)

        # handle leftover months (if not divisible evenly)
        leftover = self.n_months - len(schedule)
        if leftover > 0:
            schedule.extend([tiers[-1]] * leftover)

        return np.array(schedule, dtype=float)

    def run(self):
        all_sims = []

        for _ in range(self.n_simulations):
            self.strategy.reset()

            balance = self.start_money
            balances = []

            for month in range(self.n_months):
                balance += self.investment_schedule[month]
                balance = self.strategy.apply_return(balance)
                balances.append(balance)

            all_sims.append(balances)

        self.simulations = np.array(all_sims)

    def compute_real_balances(self):
        """
        Compute inflation-adjusted balances based on self.annual_inflation.
        Stores results in self.real_simulations.
        """
        if self.simulations is None:
            raise ValueError("Run simulations first")

        # Convert annual inflation to monthly
        monthly_inflation = (1 + self.annual_inflation)**(1/12) - 1

        # Compute cumulative inflation factor per month
        months_array = np.arange(1, self.n_months + 1)
        inflation_factor = (1 + monthly_inflation) ** months_array

        # Divide each simulation by inflation factor to get real balances
        self.real_simulations = self.simulations / inflation_factor

    def get_statistics(self):
        if self.simulations is None:
            raise ValueError("Run simulations first")
        median = np.median(self.simulations, axis=0)
        p5 = np.percentile(self.simulations, 5, axis=0)
        p25 = np.percentile(self.simulations, 25, axis=0)
        p75 = np.percentile(self.simulations, 75, axis=0)
        p95 = np.percentile(self.simulations, 95, axis=0)
        mean = np.mean(self.simulations, axis=0)
        return {"median": median, "p5": p5, "p25": p25, "p75": p75, "p95": p95, "mean": mean}

    def export_json_percentiles(self, filename_prefix="sim_percentiles"):
        """
        Export median/percentile JSON for both nominal and real balances
        """
        if self.simulations is None:
            raise ValueError("Run simulations first")

        if self.real_simulations is None:
            self.compute_real_balances()

        def compute_percentiles(data):
            median = np.median(data, axis=0)
            p5 = np.percentile(data, 5, axis=0)
            p25 = np.percentile(data, 25, axis=0)
            p75 = np.percentile(data, 75, axis=0)
            p90 = np.percentile(data, 90, axis=0)
            return {"months": list(range(1, self.n_months + 1)),
                    "median": median.tolist(),
                    "p5": p5.tolist(),
                    "p25": p25.tolist(),
                    "p75": p75.tolist(),
                    "p90": p90.tolist()}

        # Nominal
        nominal_data = compute_percentiles(self.simulations)
        with open(f"{filename_prefix}_nominal.json", "w") as f:
            json.dump(nominal_data, f)

        # Real
        real_data = compute_percentiles(self.real_simulations)
        with open(f"{filename_prefix}_real.json", "w") as f:
            json.dump(real_data, f)

        print(f"Exported nominal and real percentile JSONs with prefix {filename_prefix}")
    
    def export_json_histogram(
        self, 
        n_bad=10, 
        n_mid=2, 
        n_top=5, 
        filename_prefix="sim_histogram"
    ):
        """
        Export a custom histogram JSON of final balances with variable-width bins:
        - n_bad: number of small bins for worst-case scenarios (below 25th percentile)
        - n_mid: number of medium bins for middle scenarios (25th-75th percentile)
        - n_top: number of moderate bins for top scenarios (above 75th percentile)
        Produces both nominal and real JSON files.
        """
        if self.simulations is None:
            raise ValueError("Run simulations first!")

        if self.real_simulations is None:
            self.compute_real_balances()

        def build_hist_data(final_balances):
            # Calculate key percentiles
            q25 = np.percentile(final_balances, 25)
            q75 = np.percentile(final_balances, 75)
            min_balance = final_balances.min()
            max_balance = final_balances.max()

            # Variable-width bins
            bad_bins = np.linspace(min_balance, q25, n_bad, endpoint=False)
            mid_bins = np.linspace(q25, q75, n_mid+1, endpoint=False)  # +1 because linspace includes start
            top_bins = np.linspace(q75, max_balance, n_top+1)

            # Combine edges and ensure uniqueness
            bin_edges = np.unique(np.concatenate([bad_bins, mid_bins, top_bins]))

            # Compute histogram
            counts, bin_edges = np.histogram(final_balances, bins=bin_edges)

            # Generate readable labels
            labels = [f"${int(bin_edges[i]):,} - ${int(bin_edges[i+1]):,}" for i in range(len(bin_edges)-1)]

            return {"labels": labels, "counts": counts.tolist()}

        # Nominal
        nominal_data = build_hist_data(self.simulations[:, -1])
        with open(f"{filename_prefix}_nominal.json", "w") as f:
            json.dump(nominal_data, f)

        # Real
        real_data = build_hist_data(self.real_simulations[:, -1])
        with open(f"{filename_prefix}_real.json", "w") as f:
            json.dump(real_data, f)

        print(f"Exported nominal and real histogram JSONs with prefix {filename_prefix}")

    def export_json_sample_paths(self, n_paths=50, base_filename="sim_sample_paths"):
        if self.simulations is None:
            raise ValueError("Run simulations first")
        
        n_paths = min(n_paths, self.simulations.shape[0])
        indices = random.sample(range(self.simulations.shape[0]), n_paths)
        
        # Nominal
        paths_nominal = self.simulations[indices].tolist()
        data_nominal = {"months": list(range(1, self.n_months+1)), "paths": paths_nominal}
        filename_nom = f"{base_filename}_nominal.json"
        with open(filename_nom, "w") as f:
            json.dump(data_nominal, f)
        
        # Real (inflation-adjusted balances)
        paths_real = self.real_simulations[indices].tolist()
        data_real = {"months": list(range(1, self.n_months+1)), "paths": paths_real}
        filename_real = f"{base_filename}_real.json"
        with open(filename_real, "w") as f:
            json.dump(data_real, f)
        
        print(f"Exported {n_paths} sample paths to {filename_nom} and {filename_real}")

    def export_json_contributions(self, filename="sim_contributions.json"):
        if self.simulations is None:
            raise ValueError("Run simulations first!")

        # Median path across simulations
        median_path = np.median(self.simulations, axis=0)

        # --- FIXED: contributions now come from investment_schedule ---
        contributions = np.cumsum(self.investment_schedule) + self.start_money
        contributions = contributions.tolist()

        # Growth = portfolio minus contributions
        growth = median_path - np.array(contributions)

        # --- Export nominal ---
        data_nominal = {
            "months": list(range(1, self.n_months + 1)),
            "contributions": contributions,
            "growth": growth.tolist()
        }

        with open(filename, "w") as f:
            json.dump(data_nominal, f)

        print(f"Exported nominal contributions data to {filename}")

        # --- Export real (inflation-adjusted) ---
        if hasattr(self, "annual_inflation"):
            monthly_inflation = (1 + self.annual_inflation)**(1/12) - 1
            inflation_factor = (1 + monthly_inflation)**np.arange(1, self.n_months + 1)

            growth_real = growth / inflation_factor

            data_real = {
                "months": list(range(1, self.n_months + 1)),
                "contributions": contributions,  # still nominal
                "growth": growth_real.tolist()
            }

            filename_real = filename.replace(".json", "_real.json")
            with open(filename_real, "w") as f:
                json.dump(data_real, f)

            print(f"Exported real (inflation-adjusted) contributions data to {filename_real}")

    
def annual_to_monthly_ror(annual_return: float) -> float:
    """
    Convert an annual return (as a decimal, e.g., 0.08 for 8%) 
    to a monthly compounded return.

    Args:
        annual_return (float): Annual return as decimal (8% → 0.08)

    Returns:
        float: Monthly return as decimal (0.08 annual → ~0.006434 monthly)
    """
    return (1 + annual_return) ** (1/12) - 1

def inflation_adjusted_value(future_value, years, annual_inflation=0.02):
    """
    Convert a future nominal value into present/purchasing power.
    """
    return future_value / ((1 + annual_inflation) ** years)