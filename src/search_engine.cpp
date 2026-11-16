#include "../include/search_engine.h"
#include "../include/tokenizer.h"
#include "../include/tfidf.h"
#include "../include/document.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

const vector<Document>& SearchEngine::getDocument() const
{
    return docs;
}

const unordered_map<string, int>& SearchEngine::getDF() const
{
    return df;
}

const unordered_map<string, double>& SearchEngine::getIDF() const
{
    return idf;
}

const unordered_set<string>& SearchEngine::getStopWords() const
{
    return stopWords;
}

void SearchEngine::build(const string& folderPath, const string& stopWordsPath)
{
    stopWords.clear();
    docs.clear();
    df.clear();
    idf.clear();

    stopWords = loadStopWords(stopWordsPath);
    vector<RawFile> rawFiles = readAllTxtFiles(folderPath);

    for (const auto& rf : rawFiles)
    {
        vector<string> tokens = preprocess(rf.content, stopWords);
        if (tokens.empty())
            continue;
        Document doc;
        doc.id = static_cast<int>(docs.size());
        doc.content = rf.content;
        doc.filename = rf.filename;
        for (const auto& token : tokens)
        {
            doc.termCount[token]++;
            doc.tokenCount++;
        }
        docs.push_back(doc);
    }
    if (docs.empty())
        throw::runtime_error("\n\n\n[Error] Khong tim thay file hop le!!!!\n\n\n");
    
    df = computeDF(docs);
    idf = computeIDF(df, static_cast<int>(docs.size()));
    computeTFIDFAndNorm(docs, idf);

    isBuilt = true;
}

QueryVector SearchEngine::vectorizeQuery(const string& rawQuery) const
{
    QueryVector qVec;
    vector<string> tokens = preprocess(rawQuery, stopWords);
    if (tokens.empty())
    {
        qVec.isEmptyAfterPreprocess = true;
        return qVec;
    }
    unordered_map<string, int> qTermCount;
    int knownTokenCount = 0;

    for (const string& token : tokens)
    {
        if (idf.find(token) != idf.end())
        {
            qTermCount[token]++;
            knownTokenCount++;
        }
        else
        {
            if (find(qVec.unknownWords.begin(), qVec.unknownWords.end(), token) == qVec.unknownWords.end())
                qVec.unknownWords.push_back(token);
        }
    }

    if (knownTokenCount == 0)
    {
        return qVec;
    }
    double sumSquares = 0;
    for (const auto& [word, count] : qTermCount)
    {
        double tfval = static_cast<double>(count) / knownTokenCount;
        double tfidfval = idf.at(word) * tfval;
        qVec.weights[word] = tfidfval;
        sumSquares += tfidfval * tfidfval;
    }
    qVec.norm = sqrt(sumSquares);
    return qVec;
}

double SearchEngine::computeCosine(
    const unordered_map<string, double>& queryWeights,
    double queryNorm,
    const unordered_map<string, double>& docWeights,
    double docNorm
)
{
    if (queryNorm <= 0 || docNorm <= 0)
        return 0;
    
    double dot = 0;
    for (const auto& [word, qWeight] : queryWeights)
    {
        auto it = docWeights.find(word);
        if (it != docWeights.end())
            dot += it->second * qWeight;
    }
    return dot / (queryNorm * docNorm);
}

SearchResponse SearchEngine::searchBaseline(const string& rawQuery, int k) const
{
    if (!isBuilt)
        throw::runtime_error("\n\n\n[Error] Search engine chua duoc build!!!!\n\n\n");
    SearchResponse response;

    if (k <= 0)
        return response;
    response.queryVec = vectorizeQuery(rawQuery);
    if (response.queryVec.isEmptyAfterPreprocess || response.queryVec.norm <= 0)
        return response;
    vector<SearchResult> allResults;
    for (const auto& doc : docs)
    {
        double score = computeCosine(
            response.queryVec.weights, response.queryVec.norm,
            doc.weights,doc.norm);
        if (isfinite(score) && score > 0)
            allResults.push_back({doc.id, score});
    }
    response.totalMatched = static_cast<int>(allResults.size());
    sort(allResults.begin(), allResults.end(), [](const SearchResult& a, const SearchResult& b)
    {
        if (a.score != b.score)
            return a.score > b.score;
        return a.documentId < b.documentId;
    });

    if (static_cast<int>(allResults.size()) > k)
        allResults.resize(k);
    response.results = move(allResults);
    return response;
}