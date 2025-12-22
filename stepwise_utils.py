import numpy as np
import statsmodels.formula.api as smf

def stepwise_aic(data, response, candidate_terms, start_terms=None, direction="both", verbose=True, tol=1e-4, ):
    """
    Simple stepwise selection using AIC, roughly mimicking R's step()/stepAIC().

    Parameters
    ----------
    data : pandas.DataFrame
    Your data (e.g., Credit)
    response : str
    Dependent variable name (e.g., "Balance")
    candidate_terms : list of str
    All possible RHS terms, e.g.
    ["np.log(Income)", "np.log(Limit)", "Rating", "I(Rating**2)", "C(Cards)", ...]
    start_terms : list of str or None
    Initial model terms.
    - If None and direction is 'forward' or 'both': start with intercept only.
    - If None and direction is 'backward': start with all candidate_terms.
    direction : {"forward", "backward", "both"}
    verbose : bool
    Print each step with AIC.
    tol : float
    Minimum AIC improvement required to accept a change.

    Returns
    -------
    best_model : statsmodels OLS results
    history : list[dict] (step, terms, aic)
    """

    # ----- initialize current terms -----
    if start_terms is None:
        if direction in ["forward", "both"]:
            current_terms = []
        elif direction == "backward":
            current_terms = list(candidate_terms)
        else:
            raise ValueError("direction must be 'forward', 'backward', or 'both'")
    else:
        current_terms = list(start_terms)

    def fit_model(terms):
        if len(terms) == 0:
            formula = f"{response} ~ 1"
        else:
            rhs = " + ".join(terms)
            formula = f"{response} ~ {rhs}"
        return smf.ols(formula, data=data).fit()

    current_model = fit_model(current_terms)
    current_aic = current_model.aic

    if verbose:
        print("Initial model:")
        print(f" terms: {current_terms if current_terms else ['(Intercept only)']}")
        print(f" AIC : {current_aic:.3f}\n")

    history = [{"step": 0, "terms": list(current_terms), "aic": current_aic}]
    step = 0
    improved = True

    while improved:
        improved = False
        step += 1
        best_step_aic = current_aic
        best_step_terms = None
        best_action = None
        best_term = None

        # ---- FORWARD part ----
        if direction in ["forward", "both"]:
            remaining = [t for t in candidate_terms if t not in current_terms]
            for term in remaining:
                new_terms = current_terms + [term]
                model = fit_model(new_terms)
                aic = model.aic
                if aic + tol < best_step_aic:
                    best_step_aic = aic
                    best_step_terms = new_terms
                    best_action = "add"
                    best_term = term

        # ---- BACKWARD part ----
        if direction in ["backward", "both"] and len(current_terms) > 0:
            for term in list(current_terms):
                new_terms = [t for t in current_terms if t != term]
                model = fit_model(new_terms)
                aic = model.aic
                if aic + tol < best_step_aic:
                    best_step_aic = aic
                    best_step_terms = new_terms
                    best_action = "drop"
                    best_term = term

        if best_step_terms is not None:
            improved = True
            current_terms = best_step_terms
            current_model = fit_model(current_terms)
            current_aic = best_step_aic

            if verbose:
                action_str = "Added" if best_action == "add" else "Dropped"
                print(f"Step {step}: {action_str} {best_term}")
                print(f" terms: {current_terms}")
                print(f" AIC : {current_aic:.3f}\n")

            history.append(
                {"step": step, "terms": list(current_terms), "aic": current_aic}
            )

    if verbose:
        print("Final model:")
        print(f" terms: {current_terms}")
        print(f" AIC : {current_aic:.3f}")

    return current_model, history