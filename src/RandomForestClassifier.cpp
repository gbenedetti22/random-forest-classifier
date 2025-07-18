//
// Created by gabriele on 13/07/25.
//

#include "../include/RandomForestClassifier.h"

#include <iostream>

#include "indicators.hpp"
using namespace std;

void RandomForestClassifier::fit(const vector<vector<double>> &X, const vector<int> &y) {
    if (X.empty() || y.empty()) {
        cerr << "Cannot build the tree on dataset" << endl;
        return;
    }
    const int num_trees = trees.capacity();

    for (int i = 0; i < num_trees; i++) {
        DecisionTreeClassifier tree(params.split_criteria, params.min_samples_split, params.max_features, params.random_seed);

        if (params.bootstrap) {
            vector<vector<double>> X_sample;
            vector<int> y_sample;
            bootstrap_sample(X, y, X_sample, y_sample);

            tree.train(X_sample, y_sample);
        }else {
            tree.train(X, y);
        }
        
        trees.push_back(tree);
    }

}

int RandomForestClassifier::predict(const vector<double> &x) const {
    unordered_map<int, int> vote_counts;
    for (const auto &tree : trees) {
        int pred = tree.predict(x);
        vote_counts[pred]++;
    }
    int majority_class = -1, max_votes = -1;
    for (const auto &[label, count] : vote_counts) {
        if (count > max_votes) {
            max_votes = count;
            majority_class = label;
        }
    }
    return majority_class;
}

double RandomForestClassifier::evaluate(const vector<vector<double>> &X, const vector<int> &y) const {
    int classified = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        if (predict(X[i]) == y[i]) {
            classified++;
        }
    }

    return static_cast<double>(classified) / X.size();
}

void RandomForestClassifier::bootstrap_sample(const vector<vector<double>> &X, const vector<int> &y, vector<vector<double>> &X_sample, vector<int> &y_sample) const {
    X_sample.clear();
    y_sample.clear();

    if (params.random_seed.has_value()) {
        std::srand(params.random_seed.value());
    }else {
        std::srand(std::time(nullptr));
    }

    for (size_t i = 0; i < X.size(); ++i) {
        const int idx = rand() % X.size();

        X_sample.push_back(X[idx]);
        y_sample.push_back(y[idx]);
    }

}
