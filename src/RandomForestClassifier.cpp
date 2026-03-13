//
// Created by gabriele on 13/07/25.
//

#include "../include/RandomForestClassifier.h"

#include <cassert>
#include <iostream>
#include <omp.h>
#include <pcg32/pcg_random.hpp>

#include "Logger.hpp"

#include "utils.h"
#include <spdlog/spdlog.h>
#include <random>
#include <span>
#include "pbar.hpp"


using namespace std;

void RandomForestClassifier::fit(const vector<vector<float> > &X, const vector<int> &y, const bool transposed) {
    if (!transposed) {
        vector<float> X_train_cm;
        auto shape = transpose(X, X_train_cm);
        fit(X_train_cm, y, shape, true);
        return;
    }

    vector<float> flat = flatten(X);
    auto shape = pair{X.size(), X[0].size()};
    fit(flat, y, shape, transposed);
}

void RandomForestClassifier::fit(vector<float> &X, const vector<int> &y, pair<size_t, size_t> &shape, const bool transposed) {
    if (X.empty() || y.empty()) {
        cerr << "Cannot build the tree on dataset" << endl;
        exit(EXIT_FAILURE);
    }
    if (params.njobs == 0 || params.njobs < -1) {
        cerr << "Thread count cannot be 0 or less then -1: " << params.njobs << endl;
        exit(EXIT_FAILURE);
    }

    vector<float>& X_train = X;
    pair<size_t, size_t>& shape_X_train = shape;

    if (!transposed) {
        shape_X_train = transpose(X, X_train, shape);
    }

    num_classes = ranges::max(y) + 1;

    const int t = params.njobs;
    int threads_count = t == -1 ? omp_get_max_threads() : t;
    const int num_trees = params.n_trees;
    int master_seed = rand();

    if (params.random_seed.has_value()) {
        master_seed = *params.random_seed;
    }

    seed_seq seq{ master_seed };
    vector<uint32_t> seeds(num_trees);
    seq.generate(seeds.begin(), seeds.end());

    size_t n_samples = 0;
    if (std::holds_alternative<size_t>(params.max_samples)) {
        n_samples = std::get<size_t>(params.max_samples);
    } else {
        if (const float perc = std::get<float>(params.max_samples); perc == -1.0F) {
            n_samples = shape.first;
        }else {
            if (perc < 0.0 || perc > 1.0) {
                cerr << "max_samples (float) must be between 0 and 1" << endl;
                exit(EXIT_FAILURE);
            }
            n_samples = static_cast<size_t>(shape.first * perc);
        }
    }

    const DTreeParams dtp(
        params.split_criteria,
        params.min_samples_split,
        params.max_features,
        params.min_samples_ratio,
        params.nworkers,
        params.max_depth,
        params.max_leaf_nodes
    );

    for (int i = 0; i < num_trees; ++i) {
        trees.emplace_back(dtp, seeds[i]);
    }

    int chunks = std::max(1, num_trees / (threads_count * 4));
    auto bar = pbar::ProgressBar(num_trees);

    #pragma omp parallel for schedule(dynamic, chunks) num_threads(threads_count)
    for (int i = 0; i < num_trees; i++) {
        vector<int> indices(n_samples);
        if (params.bootstrap) {
            bootstrap_sample(n_samples, shape.first, indices);

            trees[i].train(X_train, shape_X_train, y, indices);
        } else {
            iota(indices.begin(), indices.end(), 0);
            trees[i].train(X_train, shape_X_train, y, indices);
        }

        bar.update();
    }

    bar.finish();
}

vector<int> RandomForestClassifier::predict(const vector<float> &X, const pair<size_t, size_t> &shape) const {
    const size_t n_samples = shape.first;
    const size_t n_features = shape.second;

    constexpr size_t TILE_SAMPLES = 64;
    constexpr size_t TILE_TREES = 128;

    vector all_votes(n_samples * num_classes, 0);
    const int njobs = params.njobs < 0 ? omp_get_max_threads() : params.njobs;
    const int nworkers = params.nworkers < 0 ? omp_get_max_threads() : params.nworkers;
    int threads_count = std::min(std::abs(njobs * nworkers), omp_get_max_threads());

    Logger::info("Using: {} threads for prediction", threads_count);
    size_t total_tiles = (n_samples + TILE_SAMPLES - 1) / TILE_SAMPLES;
    auto bar = pbar::ProgressBar(total_tiles);

    #pragma omp parallel for collapse(2) num_threads(threads_count)
    for (size_t ii = 0; ii < n_samples; ii += TILE_SAMPLES) {
        for (size_t tt = 0; tt < trees.size(); tt += TILE_TREES) {
            const size_t i_max = std::min(ii + TILE_SAMPLES, n_samples);
            const size_t t_max = std::min(tt + TILE_TREES, trees.size());

            for (size_t i = ii; i < i_max; ++i) {
                const float* sample_ptr = X.data() + i * n_features;

                for (size_t tree = tt; tree < t_max; ++tree) {
                    const int pred = trees[tree].predict(sample_ptr);

                    #pragma omp atomic
                    all_votes[i * num_classes + pred]++;
                }
            }

            if (tt + TILE_TREES >= trees.size()) {
                bar.update(i_max - ii);
            }
        }
    }
    bar.finish();

    vector predictions(n_samples, 0);

    #pragma omp parallel for num_threads(threads_count)
    for (int sample_id = 0; sample_id < n_samples; ++sample_id) {
        auto vote_counts = span(all_votes.data() + sample_id * num_classes, num_classes);
        int max_votes = -1;
        int majority_class = -1;

        for (int label = 0; label < num_classes; label++) {
            if (vote_counts[label] > max_votes) {
                max_votes = vote_counts[label];
                majority_class = label;
            }
        }

        predictions[sample_id] = majority_class;
    }

    return std::move(predictions);
}

pair<float, float> RandomForestClassifier::score(const std::vector<std::vector<float>> &X, const std::vector<int> &y) const {
    return score(flatten(X), y, pair{X.size(), X[0].size()});
}

pair<float, float> RandomForestClassifier::score(const vector<float> &X, const vector<int> &y, const pair<size_t, size_t>& shape) const {
    // const int njobs = params.njobs < 0 ? omp_get_max_threads() : params.njobs;
    // const int nworkers = params.nworkers < 0 ? omp_get_max_threads() : params.nworkers;
    // const int threads_count = std::min(std::abs(njobs * nworkers), omp_get_max_threads());
    //
    // cout << "Threads used for prediction: " << threads_count << endl;
    // const auto& all_votes = FarmFF::getVotes(X, shape, trees, threads_count, num_classes);
    const auto& all_votes = predict(X, shape);

    return make_pair(accuracy(y, all_votes), f1_score(y, all_votes));
}

void RandomForestClassifier::bootstrap_sample(const size_t n_samples,
                                              const size_t total_features,
                                              std::vector<int> &indices) const
{
    thread_local pcg32 rng(std::random_device{}());
    thread_local bool initialized = false;

    if (params.random_seed.has_value() && !initialized) {
        rng.seed(params.random_seed.value() + omp_get_thread_num());
        initialized = true;
    }

    for (size_t i = 0; i < n_samples; ++i) {
        indices[i] = rng(total_features);
    }
}

