#pragma once
#include <string>
#include <vector>
#include <unordered_set>

using namespace std;

unordered_set<string> loadStopWords(const string& filepath);
vector<string> tokenize(const string& text);
vector<string> preprocess(const string& text, const unordered_set<string>& stopWords);