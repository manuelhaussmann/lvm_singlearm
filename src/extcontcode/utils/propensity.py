from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression


def fit_psmodel(data, treatment, visualize=False, title="None", save_str=None):
    psmodel = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5)
    psmodel.fit(data, treatment)
    ps_estimate = psmodel.predict_proba(data)[:, 1]
    if visualize:
        plt.hist(ps_estimate[treatment.eq(0)], alpha=0.8, density=True, label="control")
        plt.hist(
            ps_estimate[treatment.eq(1)], alpha=0.8, density=True, label="treatment"
        )
        plt.legend()
        plt.title(title)
        if save_str is not None:
            plt.savefig(save_str)
            plt.close()
        else:
            plt.show()
    return ps_estimate
