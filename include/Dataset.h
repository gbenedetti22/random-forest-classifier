//
// Created by gabriele on 13/07/25.
//

#ifndef DATASET_H
#define DATASET_H
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

class Dataset {
public:
    Dataset();

    static std::pair<std::vector<std::vector<double>>, std::vector<int>> load_classification(const std::string &filename);

    static std::tuple<std::vector<std::vector<double>>, std::vector<int>, std::vector<std::vector<double>>, std::vector<int>> train_test_split(std::vector<std::vector<double>> &X, std::vector<int> &y, double train_perc = 0.8, bool stratified = false);

private:
    static std::vector<std::string> split(const std::string &s, char delim = ',');
    static std::pair<std::vector<std::vector<double>>, std::vector<int>> load_iris(const std::string &filename);
    static std::pair<std::vector<std::vector<double>>, std::vector<int>> load_susy(const std::string &filename);

    static void shuffle_data(std::vector<std::vector<double>> &X, std::vector<int> &y);
};

#endif //DATASET_H
