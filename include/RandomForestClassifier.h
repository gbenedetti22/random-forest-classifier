//
// Created by gabriele on 13/07/25.
//

#ifndef RANDOMFORESTCLASSIFIER_H
#define RANDOMFORESTCLASSIFIER_H
#include <optional>
#include <variant>
#include <vector>
#include "DecisionTreeClassifier.h"

class RandomForestClassifier {
public:
    struct RandomForestParams {
        int n_trees = 10;
        std::string split_criteria = "gini";
        int min_samples_split = 2;
        const std::variant<int, std::string> max_features = "sqrt";
        bool bootstrap = true;
        const std::optional<int> random_seed = std::nullopt;
    };

    explicit RandomForestClassifier(const RandomForestParams &params)
        : params(params) {
        trees.reserve(params.n_trees);
    }

    void fit(const std::vector<std::vector<double> > &X, const std::vector<int> &y);

    int predict(const std::vector<double> &x) const;

    double evaluate(const std::vector<std::vector<double> > &X, const std::vector<int> &y) const;

private:
    RandomForestParams params;
    std::vector<DecisionTreeClassifier> trees;

    void bootstrap_sample(const std::vector<std::vector<double> > &X, const std::vector<int> &y,
                                 std::vector<std::vector<double> > &X_sample, std::vector<int> &y_sample) const;
};


#endif //RANDOMFORESTCLASSIFIER_H
