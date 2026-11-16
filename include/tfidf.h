#pragma once
#include "document.h"
#include <string>
#include <vector>
#include <unordered_map>

using namespace std;

unordered_map<string, int> computeDF(vector<Document>& docs);
unordered_map<string, double> computeIDF(
    const unordered_map<string, int>& df,
    int totalDocsN
);
void computeTFIDFAndNorm(
    vector<Document>& docs,
    const unordered_map<string, double>& idf
);
