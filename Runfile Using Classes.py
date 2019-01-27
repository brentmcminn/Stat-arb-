# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:49:43 2019

@author: Brent
"""

# =============================================================================
# Import packages
# =============================================================================

import sys
import cvxpy as cvx
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os

# =============================================================================
# Import data
# =============================================================================
#homedir = os.environ['HOMEPATH']
wdir = '/Users/brentmcminn/Downloads/wetransfer-45b7f8'#os.path.join(homedir , 'Dropbox/Data/Equity')


Close = pd.read_csv(wdir+'/adjClose.csv',
                    index_col = 0,
                    parse_dates = True)

Open = pd.read_csv(wdir+'/adjOpen.csv',
                    index_col = 0,
                    parse_dates = True)

SPX = pd.read_csv(wdir+'/SP500.csv',
                    index_col = 0,
                    parse_dates = True)
SPX = SPX.reindex(Close.index)

Historical_Constituents = pd.read_csv(wdir+'/Historical Constituents.csv',
                                      index_col = 0,
                                      parse_dates = True)
Historical_Constituents['PX'] = np.nan

Historical_Constituents = Historical_Constituents.reindex(Close.index)

Historical_Constituents = Historical_Constituents[Close.columns]

returns = Close.iloc[-1250:].pct_change()[1:]

Current_Constituents = Historical_Constituents.columns[Historical_Constituents
                                                       .iloc[-1] == 1]

returns = returns[Current_Constituents]

#set > 10 s.d. returns to nan
sd_limit = 10
mask = abs(returns.subtract(returns.mean(axis = 1), axis = 0)
            ).gt(sd_limit * returns.std(axis = 1), axis = 0)
returns[mask] = np.nan
returns[np.isnan(returns)] = 0

five_year_returns = returns
universe_tickers = five_year_returns.columns.values


# =============================================================================
# Returns reasonability check
# =============================================================================

plt.plot(five_year_returns.quantile(0.25), label = 'Q25')
plt.plot(five_year_returns.quantile(0.5), label = 'Q50')
plt.plot(five_year_returns.quantile(0.75), label = 'Q75')
plt.legend()
plt.show()

# # Statistical Risk Model
# It's time to build the risk model. You'll be creating a statistical risk model using PCA. So, the first thing is building the PCA model.
# ## Fit PCA

from sklearn.decomposition import PCA


def fit_pca(returns, num_factor_exposures, svd_solver):
    """
    Fit PCA model with returns.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    num_factor_exposures : int
        Number of factors for PCA
    svd_solver: str
        The solver to use for the PCA model

    Returns
    -------
    pca : PCA
        Model fit to returns
    """
    pca = PCA(n_components = num_factor_exposures, svd_solver = svd_solver)
    pca.fit(returns)
    
    return pca

# Let's see what the model looks like. First, we'll look at the PCA components.

# In[10]:


num_factor_exposures = 20
pca = fit_pca(five_year_returns, num_factor_exposures, 'full')

pca.components_


# Let's also look at the PCA's percent of variance explained by each factor

# In[11]:


plt.bar(np.arange(num_factor_exposures), pca.explained_variance_ratio_)


# You will see that the first factor dominates. The precise definition of each factor in a latent model is unknown, however we can guess at the likely interpretation.

# ## Factor Betas
# Implement `factor_betas` to get the factor betas from the PCA model.

# In[12]:


def factor_betas(pca, factor_beta_indices, factor_beta_columns):
    """
    Get the factor betas from the PCA model.

    Parameters
    ----------
    pca : PCA
        Model fit to returns
    factor_beta_indices : 1 dimensional Ndarray
        Factor beta indices
    factor_beta_columns : 1 dimensional Ndarray
        Factor beta columns

    Returns
    -------
    factor_betas : DataFrame
        Factor betas
    """
    assert len(factor_beta_indices.shape) == 1
    assert len(factor_beta_columns.shape) == 1
        
    return pd.DataFrame(pca.components_.T, 
                        factor_beta_indices, 
                        factor_beta_columns)



# ### View Data
# Let's view the factor betas from this model.

# In[13]:


risk_model = {}
risk_model['factor_betas'] = factor_betas(pca, 
                                          five_year_returns.columns.values, 
                                          np.arange(num_factor_exposures))

risk_model['factor_betas']


# ## Factor Returns
# Implement `factor_returns` to get the factor returns from the PCA model using the returns data.

# In[14]:


def factor_returns(pca, returns, factor_return_indices, factor_return_columns):
    """
    Get the factor returns from the PCA model.

    Parameters
    ----------
    pca : PCA
        Model fit to returns
    returns : DataFrame
        Returns for each ticker and date
    factor_return_indices : 1 dimensional Ndarray
        Factor return indices
    factor_return_columns : 1 dimensional Ndarray
        Factor return columns

    Returns
    -------
    factor_returns : DataFrame
        Factor returns
    """
    assert len(factor_return_indices.shape) == 1
    assert len(factor_return_columns.shape) == 1
        
    return pd.DataFrame(pca.transform(returns), 
                        factor_return_indices, 
                        factor_return_columns)


# ### View Data
# Let's see what these factor returns looks like over time.

# In[15]:


risk_model['factor_returns'] = factor_returns(
    pca,
    five_year_returns,
    five_year_returns.index,
    np.arange(num_factor_exposures))

risk_model['factor_returns'].cumsum().plot(legend=None)


# ## Factor Covariance Matrix
# Implement `factor_cov_matrix` to get the factor covariance matrix.

# In[16]:


def factor_cov_matrix(factor_returns, ann_factor):
    """
    Get the factor covariance matrix

    Parameters
    ----------
    factor_returns : DataFrame
        Factor returns
    ann_factor : int
        Annualization factor

    Returns
    -------
    factor_cov_matrix : 2 dimensional Ndarray
        Factor covariance matrix
    """
        
    return np.diag(factor_returns.var(axis = 0, ddof = 1)* ann_factor)
ann_factor = 252
risk_model['factor_cov_matrix'] = factor_cov_matrix(risk_model['factor_returns'], ann_factor)

risk_model['factor_cov_matrix']


# ## Idiosyncratic Variance Matrix
# Implement `idiosyncratic_var_matrix` to get the idiosyncratic variance matrix.

def idiosyncratic_var_matrix(returns, factor_returns, factor_betas, ann_factor):
    """
    Get the idiosyncratic variance matrix

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    factor_returns : DataFrame
        Factor returns
    factor_betas : DataFrame
        Factor betas
    ann_factor : int
        Annualization factor

    Returns
    -------
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix
    """
    
    common_returns = pd.DataFrame(np.dot(factor_returns, factor_betas.T), returns.index, returns.columns)
    residuals = (returns- common_returns)
    
    return pd.DataFrame(np.diag(np.var(residuals))*ann_factor, returns.columns, returns.columns)
# ### View Data


risk_model['idiosyncratic_var_matrix'] = idiosyncratic_var_matrix(five_year_returns, risk_model['factor_returns'], risk_model['factor_betas'], ann_factor)

risk_model['idiosyncratic_var_matrix']

def idiosyncratic_var_vector(returns, idiosyncratic_var_matrix):
    """
    Get the idiosyncratic variance vector

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix

    Returns
    -------
    idiosyncratic_var_vector : DataFrame
        Idiosyncratic variance Vector
    """
    
    result = pd.DataFrame(np.diagonal(idiosyncratic_var_matrix),index = returns.columns)
    
    return result

risk_model['idiosyncratic_var_vector'] = idiosyncratic_var_vector(five_year_returns, risk_model['idiosyncratic_var_matrix'])

risk_model['idiosyncratic_var_vector']


# ## Predict using the Risk Model
# Using the data we calculated in the risk model, implement `predict_portfolio_risk` to predict the portfolio risk using the formula $ \sqrt{X^{T}(BFB^{T} + S)X} $ where:
# - $ X $ is the portfolio weights
# - $ B $ is the factor betas
# - $ F $ is the factor covariance matrix
# - $ S $ is the idiosyncratic variance matrix


def predict_portfolio_risk(factor_betas, factor_cov_matrix, idiosyncratic_var_matrix, weights):
#    factor_betas = risk_model['factor_betas']
#    factor_cov_matrix = risk_model['factor_cov_matrix']
#    idiosyncratic_var_matrix = risk_model['idiosyncratic_var_matrix']
    """
    Get the predicted portfolio risk
    
    Formula for predicted portfolio risk is sqrt(X.T(BFB.T + S)X) where:
      X is the portfolio weights
      B is the factor betas
      F is the factor covariance matrix
      S is the idiosyncratic variance matrix

    Parameters
    ----------
    factor_betas : DataFrame
        Factor betas
    factor_cov_matrix : 2 dimensional Ndarray
        Factor covariance matrix
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix
    weights : DataFrame
        Portfolio weights

    Returns
    -------
    predicted_portfolio_risk : float
        Predicted portfolio risk
    """
    assert len(factor_cov_matrix.shape) == 2
    
    result = np.sqrt(weights.T.dot(factor_betas.dot(factor_cov_matrix.dot(factor_betas.T))+
                                   idiosyncratic_var_matrix.values).dot(weights.values))
    return result.values[0][0]


all_weights = pd.DataFrame(np.repeat(1/len(universe_tickers),
                                     len(universe_tickers)), 
                            universe_tickers)

predict_portfolio_risk(
    risk_model['factor_betas'],
    risk_model['factor_cov_matrix'],
    risk_model['idiosyncratic_var_matrix'],
    all_weights)

# # Create Alpha Factors
# With the profile risk calculated, it's time to start working on the alpha factors. In this project, we'll create the following factors:
# - Momentum 1 Year Factor
# - Mean Reversion 5 Day Sector Neutral Factor
# - Mean Reversion 5 Day Sector Neutral Smoothed Factor
# - Overnight Sentiment Factor
# - Overnight Sentiment Smoothed Factor
# 
# ## Momentum 1 Year Factor
# Each factor will have a hypothesis that goes with it. For this factor, it is "Higher past 12-month (252 days) returns are proportional to future return." Using that hypothesis, we've generated this code:

def standardize(df):
#    prices = Close
#    constituents = universe_tickers
#    mom = (prices/prices.shift(252) - 1).shift(20)[constituents]
#    df = mom
#    #df values = (df - mean(day)/std(day)
#    df.mean(axis = 1)
#    df.std(ddof = 1, axis = 1)
#    df.subtract(df.mean(axis = 1), axis = 0)
    result = df.subtract(df.mean(axis = 1), axis = 0).divide(
    df.std(ddof = 1, axis = 1), axis = 0)
    return result

def momentum_1yr(prices, constituents):
    mom = (prices/prices.shift(252) - 1).shift(20)[constituents]
    mom = standardize(mom)
    return mom.rolling(252).mean()
    #mom = mom.rank()


def mean_reversion_5day_smoothed(prices, constituents):
    mr = (prices/prices.shift(5) - 1)[constituents]
    mr = standardize(mr)
    return -1* mr.rolling(5).mean()

def overnight_sentiment(closes, opens, constituents):
#    closes = Close
#    opens = Open
#    constituents = universe_tickers
    """
    Computes the overnight return, per hypothesis from
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554010
    """
    CTO = (opens.subtract(closes.shift(1)).divide(closes.shift(1)))
    CTO = CTO[constituents]
    TOR = CTO.rolling(20).sum() #TrailingOvernightReturns
    overnight_sentiment = standardize(TOR)
    return overnight_sentiment.rolling(5).mean()

Momentum_1YR = momentum_1yr(Close, universe_tickers)
Mean_Reversion_5Day_Smoothed = mean_reversion_5day_smoothed(Close, 
                                                            universe_tickers)
Overnight_Sentiment_Smoothed = overnight_sentiment(Close,
                                                   Open, 
                                                   universe_tickers)
import seaborn as sns
sns.kdeplot(overnight_sentiment.iloc[-1])

def quantize(df):
    #long short, gross exposure = 1, net exposure = 0
    ''' 
    weights are computed by demeaning factor values and dividing by the sum
    of their absolute value (achieving gross leverage of 1). The sum of
    positive weights will be the same as the negative weights (absolute
    value), suitable for a dollar neutral long-short portfolio'''

    df = df.subtract(df.mean(axis = 1), axis = 0)
    df = df.divide(df.abs().sum(axis = 1), axis = 0)
    return df

Momentum_1YR_weights = quantize(Momentum_1YR)
Mean_Reversion_5Day_Smoothed_weights = quantize(Mean_Reversion_5Day_Smoothed)
Overnight_Sentiment_Smoothed_weights = quantize(Overnight_Sentiment_Smoothed)

def alpha_return(weights, returns):
    weighted_rets = weights.shift(1).multiply(returns)
    factor_ret = weighted_rets.sum(axis = 1)
    return factor_ret

ls_factor_returns = pd.DataFrame()
ls_factor_returns['Momentum_1YR'] = alpha_return(Momentum_1YR_weights, returns)
ls_factor_returns['Mean_Reversion_5D'] = \
                alpha_return(Mean_Reversion_5Day_Smoothed_weights,returns)

ls_factor_returns['Overnight_Sentiment_Smoothed'] = \
                alpha_return(Overnight_Sentiment_Smoothed_weights, returns)

(1+ls_factor_returns).cumprod().plot()
plt.show()

#Mean returns by quantile
def bucket_factor(df):
    out = np.zeros(np.shape(df))
    for k in range(len(df)):
        data = df.iloc[k]
        try:
            out[k,:] = (pd.qcut(data,5, labels = False) + 1).values
        except ValueError:
            out[k,:] = np.nan
    out = pd.DataFrame(out, index = df.index, columns = df.columns)
    return out
    
def quantile_bar_plot(factor_values, returns):
    bucketed = bucket_factor(factor_values)
    out = pd.Series()
    for k in range(1,6):
        out.loc[k] = np.nanmean(returns[bucketed == k])
    return out

qr_factor_returns = pd.DataFrame()
qr_factor_returns['Momentum_1YR'] = quantile_bar_plot(Momentum_1YR, returns)
qr_factor_returns['Mean_Reversion_5Day'] = quantile_bar_plot(Mean_Reversion_5Day_Smoothed,
                                                    returns)
qr_factor_returns['Overnight_Sentiment_Smoothed'] = quantile_bar_plot(Overnight_Sentiment_Smoothed, returns)
(10000*qr_factor_returns).plot.bar(
    subplots=True,
    sharey=True,
    layout=(4,2),
    figsize=(14, 14),
    legend=False)
plt.show()

def factor_rank_autocorrelation(df, period=1):
    """
    Computes autocorrelation of mean factor ranks in specified time spans.
    We must compare period to period factor ranks rather than factor values
    to account for systematic shifts in the factor values of all names or names
    within a group. This metric is useful for measuring the turnover of a
    factor. If the value of a factor for each name changes randomly from period
    to period, we'd expect an autocorrelation of 0.

    """
    npranks = df.rank(axis = 0).values
    out = np.zeros((np.shape(df)[0],1))
    for k in range(1,len(npranks)):       
        no_nan_mask = np.where(~np.isnan(npranks[k,:]))[0]
        out[k,:] = np.corrcoef(npranks[k-1,no_nan_mask],
                               npranks[k,no_nan_mask])[1,0]
    out = pd.DataFrame(out, 
                       index = df.index, 
                       columns = ['Factor_rank_Autocorr'])
    out[out == 0] = np.nan
    return out

ls_FRA = factor_rank_autocorrelation(Momentum_1YR, returns)

ls_FRA['Mean_Reversion_5Day'] = factor_rank_autocorrelation(Mean_Reversion_5Day_Smoothed,
                                                    returns)['Factor_rank_Autocorr']
ls_FRA['Overnight_Sentiment_Smoothed'] = factor_rank_autocorrelation(Overnight_Sentiment_Smoothed,
                                                    returns)['Factor_rank_Autocorr']

ls_FRA.plot(title = 'Factor Rank Autocorrelation')

def sharpe_ratio(factor_returns, annualization_factor):
    """
    Get the sharpe ratio for each factor for the entire period

    Parameters
    ----------
    factor_returns : DataFrame
        Factor returns for each factor and date
    annualization_factor: float
        Annualization Factor

    Returns
    -------
    sharpe_ratio : Pandas Series of floats
        Sharpe ratio
    """    
    mean = factor_returns.mean()
    std = factor_returns.std(ddof = 1)
    return (mean / std) * annualization_factor

daily_annualization_factor = np.sqrt(252)
sharpe_ratio(ls_factor_returns, daily_annualization_factor).round(2)

alpha_vector = pd.DataFrame((Momentum_1YR \
                + Mean_Reversion_5Day_Smoothed \
                + Overnight_Sentiment_Smoothed).iloc[-1].values,
                index = universe_tickers, 
                columns = ['alpha_vector'])
alpha_vector[pd.isnull(alpha_vector)] = 0
                
from abc import ABC, abstractmethod


class AbstractOptimalHoldings(ABC):    
    @abstractmethod
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        
        raise NotImplementedError()
    
    @abstractmethod
    def _get_constraints(self, weights, factor_betas, risk):
        """
        Get the constraints

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        factor_betas : 2 dimensional Ndarray
            Factor betas
        risk: CVXPY Atom
            Predicted variance of the portfolio returns

        Returns
        -------
        constraints : List of CVXPY Constraint
            Constraints
        """
        
        raise NotImplementedError()
        
    def _get_risk(self, weights, factor_betas, alpha_vector_index, factor_cov_matrix, idiosyncratic_var_vector):
        f = factor_betas.loc[alpha_vector_index].values.T * weights
        X = factor_cov_matrix
        S = np.diag(idiosyncratic_var_vector.loc[alpha_vector_index].values.flatten())

        return cvx.quad_form(f, X) + cvx.quad_form(weights, S)
    
    def find(self, alpha_vector, factor_betas, factor_cov_matrix, idiosyncratic_var_vector):
        weights = cvx.Variable(len(alpha_vector))
        risk = self._get_risk(weights, factor_betas, alpha_vector.index, factor_cov_matrix, idiosyncratic_var_vector)
        
        obj = self._get_obj(weights, alpha_vector)
        constraints = self._get_constraints(weights, factor_betas.loc[alpha_vector.index].values, risk)
        
        prob = cvx.Problem(obj, constraints)
        prob.solve(max_iters=500)

        optimal_weights = np.asarray(weights.value).flatten()
        
        return pd.DataFrame(data=optimal_weights, index=alpha_vector.index)
    
class OptimalHoldings(AbstractOptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        assert(len(alpha_vector.columns) == 1)

        return cvx.Maximize(alpha_vector.values.T @ weights)
    
    def _get_constraints(self, weights, factor_betas, risk):
        """
        Get the constraints

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        factor_betas : 2 dimensional Ndarray
            Factor betas
        risk: CVXPY Atom
            Predicted variance of the portfolio returns

        Returns
        -------
        constraints : List of CVXPY Constraint
            Constraints
        """
        assert(len(factor_betas.shape) == 2)
        
        constraints = [risk <= self.risk_cap ** 2,
                       factor_betas.T * weights <= self.factor_max,
                       factor_betas.T * weights >= self.factor_min,
                      cvx.sum(weights) == 0,
                      sum(cvx.abs(weights)) <= 1,
                       cvx.min(weights) >= self.weights_min,
                       cvx.max(weights) <= self.weights_max]
        
        return constraints 

    def __init__(self, risk_cap=0.05, factor_max=10.0, factor_min=-10.0, weights_max=0.55, weights_min=-0.55):
        self.risk_cap=risk_cap
        self.factor_max=factor_max
        self.factor_min=factor_min
        self.weights_max=weights_max
        self.weights_min=weights_min


optimal_weights = OptimalHoldings().find(alpha_vector, 
                                         risk_model['factor_betas'], 
                                         risk_model['factor_cov_matrix'], 
                                         risk_model['idiosyncratic_var_vector'])

optimal_weights.plot.bar(legend=None, title='Portfolio % Holdings by Stock')
x_axis = plt.axes().get_xaxis()
x_axis.set_visible(False)

def get_factor_exposures(factor_betas, weights):
    return factor_betas.loc[weights.index].T.dot(weights)

get_factor_exposures(risk_model['factor_betas'], optimal_weights).plot.bar(
    title='Portfolio Net Factor Exposures',
    legend=False)

class OptimalHoldingsRegualization(OptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        assert(len(alpha_vector.columns) == 1)
        
        return cvx.Maximize(alpha_vector.values.T @ weights 
                            - self.lambda_reg 
                            * cvx.norm(weights, 2))
        

    def __init__(self, lambda_reg=0.5, risk_cap=0.05, factor_max=10.0, factor_min=-10.0, weights_max=0.55, weights_min=-0.55):
        self.lambda_reg = lambda_reg
        self.risk_cap=risk_cap
        self.factor_max=factor_max
        self.factor_min=factor_min
        self.weights_max=weights_max
        self.weights_min=weights_min


optimal_weights_1 = OptimalHoldingsRegualization(lambda_reg=5.0).find(alpha_vector, risk_model['factor_betas'], risk_model['factor_cov_matrix'], risk_model['idiosyncratic_var_vector'])

optimal_weights_1.plot.bar(legend=None, title='Portfolio % Holdings by Stock')
x_axis = plt.axes().get_xaxis()
x_axis.set_visible(False)

get_factor_exposures(risk_model['factor_betas'], optimal_weights_1).plot.bar(
    title='Portfolio Net Factor Exposures',
    legend=False)

class OptimalHoldingsStrictFactor(OptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        assert(len(alpha_vector.columns) == 1)
        
        alpha_bar = np.mean(alpha_vector)
        denom = np.sum(abs(alpha_vector))
        x_star = ((alpha_vector - alpha_bar) / denom)
        return cvx.Minimize(cvx.norm(weights 
                                     - cvx.reshape(x_star,weights.shape), 2))

optimal_weights_2 = OptimalHoldingsStrictFactor(
    weights_max=0.02,
    weights_min=-0.02,
    risk_cap=0.0015,
    factor_max=0.015,
    factor_min=-0.015).find(alpha_vector, risk_model['factor_betas'], risk_model['factor_cov_matrix'], risk_model['idiosyncratic_var_vector'])

optimal_weights_2.plot.bar(legend=None, title='Portfolio % Holdings by Stock')
x_axis = plt.axes().get_xaxis()
x_axis.set_visible(False)

get_factor_exposures(risk_model['factor_betas'], optimal_weights_2).plot.bar(
    title='Portfolio Net Factor Exposures',
    legend=False)