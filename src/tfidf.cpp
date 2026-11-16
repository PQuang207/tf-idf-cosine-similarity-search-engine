#include "../include/tfidf.h"
#include <cmath>
#include <stdexcept>
using namespace std;

unordered_map<string, int> computeDF(vector<Document>& docs)
{
    unordered_map<string, int> df;
    for (const auto& doc : docs)
        for (const auto& [word, count] : doc.termCount)
            df[word] += 1;
    return df;
}

unordered_map<string, double> computeIDF(
    const unordered_map<string, int>& df,
    int totalDocsN
)
{
    unordered_map<string, double> idf;
    if (totalDocsN <= 0)
        return idf;
    double Ndouble = static_cast<double>(totalDocsN);
    for (const auto& [word, docFreq] : df)
        idf[word] = log((Ndouble + 1) / (static_cast<double>(docFreq) + 1)) + 1;
    return idf;
}

void computeTFIDFAndNorm(
    vector<Document>& docs,
    const unordered_map<string, double>& idf
)
{
    for (auto& doc : docs)
    {
        doc.weights.clear();
        doc.norm = 0;
        
        if (doc.tokenCount <= 0)
            continue;
        double sumSquares = 0;
        for (const auto& [word, count] : doc.termCount)
        {
            double tf = static_cast<double>(count) / static_cast<double>(doc.tokenCount);
            double wordIdf = idf.at(word);
            double weight = tf * wordIdf;
            doc.weights[word] = weight;
            sumSquares += weight * weight;
        }
        doc.norm = sqrt(sumSquares);
    }
}

