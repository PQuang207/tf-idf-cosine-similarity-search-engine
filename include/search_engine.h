#pragma once
#include <unordered_map>
#include <string>
#include <vector>
#include "document.h"
#include <unordered_set>

struct QueryVector
{
    unordered_map<string, double> weights;
    double norm = 0;
    vector<string> unknownWords;
    bool isEmptyAfterPreprocess = false;
};

struct SearchResult
{
    int documentId = -1;
    double score = 0;
};

struct SearchResponse
{
    QueryVector queryVec;
    int totalMatched = 0;
    vector<SearchResult> results;
};

class SearchEngine
{
private:
    bool isBuilt = false;
    unordered_set<string> stopWords;
    vector<Document> docs;
    unordered_map<string, int> df;
    unordered_map<string, double> idf;
public:
    void build(const string& folderPath, const string& stopWordsPath);
    QueryVector vectorizeQuery(const string& rawQuery) const;
    
    static double computeCosine(
        const unordered_map<string, double>& queryWeights,
        double queryNorm,
        const unordered_map<string, double>& docWeights,
        double docNorm
    );
    SearchResponse searchBaseline(const string& rawQuery, int k = 5) const;
    const vector<Document>& getDocument() const;
    const unordered_map<string, int>& getDF() const;
    const unordered_map<string, double>& getIDF() const;
    const unordered_set<string>& getStopWords() const;
};
