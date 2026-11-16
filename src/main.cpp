#include "../include/document.h"
#include "../include/tokenizer.h"
#include "../include/tfidf.h"
#include "search_engine.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <filesystem>

using namespace std;

struct CorpusState
{
    unordered_set<string> stopWords;
    vector<Document> docs;
    unordered_map<string, int> df;
    unordered_map<string, double> idf;
};

void buildCorpus(const string& folderPath, const string& stopWordsPath, CorpusState& state);
bool isClose(double actual, double expected, double eps = 1e-6);
void testPhase1(const unordered_set<string>& stopWords);
void testPhase2(const CorpusState& state);
void testPhase12(const SearchEngine& engine);
void testPhase3(const SearchEngine& engine);
string resolvePath(const string& relativePath);
void printSearchResponse(const SearchEngine& engine, const string& query, int k);

int main()
{
    SearchEngine engine;
    engine.build(resolvePath("data/toy"), resolvePath("stopWords.txt"));
    engine.build(resolvePath("data/toy"), resolvePath("stopWords.txt"));
    testPhase12(engine);
    testPhase3(engine);
    printSearchResponse(engine, "machine learning", 5);
    return 0;
}

string resolvePath(const string& relativePath)
{
    if (filesystem::exists(relativePath))
        return relativePath;
    return "../" + relativePath;
}

bool isClose(double actual, double expected, double eps)
{
    return abs(actual - expected) < eps;
}

void buildCorpus(const string& folderPath, const string& stopWordsPath, CorpusState& state)
{
    state.stopWords.clear();
    state.docs.clear();
    state.df.clear();
    state.idf.clear();
    state.stopWords = loadStopWords(stopWordsPath);
    vector<RawFile> rawFiles = readAllTxtFiles(folderPath);
    for (const auto& rf : rawFiles)
    {
        vector<string> tokens = preprocess(rf.content, state.stopWords);
        if (tokens.empty())
        {
            continue;
        }
        Document doc;
        doc.id = state.docs.size();
        doc.filename = rf.filename;
        doc.content = rf.content;
        for (const auto& token: tokens)
        {
            doc.termCount[token]++;
            doc.tokenCount++;
        }
        state.docs.push_back(doc);
    }
    if (state.docs.empty())
        throw::runtime_error("\n\n\n[Error] Khong tim thay file hop le!!!\n\n");
    state.df = computeDF(state.docs);
    state.idf = computeIDF(state.df, static_cast<int>(state.docs.size()));
    computeTFIDFAndNorm(state.docs, state.idf);
}

void testPhase1(const unordered_set<string>& stopWords)
{
    cout << "=== CHAY KIEM THU CHANG 1 ===\n";
    auto t1 = tokenize("Hello, MACHINE learning!!!");
    assert((t1 == vector<string>{"hello", "machine", "learning"}));

    auto t2 = tokenize("C++ and C#");
    assert((t2 == vector<string>{"c++", "and", "c#"}));

    auto t3 = tokenize("data-base");
    // for (const string& x: t3)
    //     cout << x << "\n";
    assert((t3 == vector<string>{"data", "base"}));

    auto t4 = tokenize("      \n\t    ");
    assert(t4.empty());

    auto p1 = preprocess("machine learning is a branch of artificial intelligence", stopWords);
    assert((p1 == vector<string>{"machine", "learning", "branch", "artificial", "intelligence"}));
    assert(p1.size() == 5);
    cout <<"\n\n\n[PASSED] Da vuot qua tat ca cac testcases chang 1!!!!\n\n\n";
}

void testPhase2(const CorpusState& state)
{
    int n = static_cast<int>(state.docs.size());
    assert(n == 3);
    for (const auto& [word, freq] : state.df)
    {
        assert(freq >= 1 && freq <= n);
        double idfval = state.idf.at(word);
        assert(isfinite(idfval) && idfval >= 1);
    }

    assert(state.df.at("machine") == 2);
    assert(state.df.at("learning") == 1);
    assert(state.df.at("vision") == 1);
    assert(state.df.at("python") == 1);
    assert(state.df.at("code") == 1);

    assert(isClose(state.idf.at("machine"), 1.287682072));
    assert(isClose(state.idf.at("learning"), 1.693147181));
    assert(isClose(state.idf.at("vision"), 1.693147181));
    assert(isClose(state.idf.at("python"), 1.693147181));
    assert(isClose(state.idf.at("code"), 1.693147181));

    const Document& d1 = state.docs[0];
    assert(isClose(d1.weights.at("machine"), 0.858454715));
    assert(isClose(d1.weights.at("learning"), 0.564382394));
    assert(isClose(d1.norm, 1.027362));

    const Document& d2 = state.docs[1];
    assert(isClose(d2.weights.at("machine"), 0.6438410362));
    assert(isClose(d2.weights.at("vision"), 0.8465735903));
    assert(isClose(d2.norm, 1.063587384));

    const Document& d3 = state.docs[2];
    assert(isClose(d3.weights.at("python"), 0.8465735903));
    assert(isClose(d3.weights.at("code"), 0.8465735903));
    assert(isClose(d3.norm, 1.197235853));

    cout <<"\n\n\n[PASSED] Da vuot qua tat ca cac testcase chang 2!!!!\n\n\n";
}

void testPhase12(const SearchEngine& engine)
{
    const auto& docs = engine.getDocument();
    const auto& df = engine.getDF();
    const auto& idf = engine.getIDF();

    assert(docs.size() == 3);
    assert(df.at("machine") == 2);
    assert(isClose(idf.at("machine"), 1.287682072));
    assert(isClose(idf.at("learning"), 1.693147181));
    assert(isClose(docs[0].norm, 1.027362));
    assert(isClose(docs[1].norm, 1.063587));
    assert(isClose(docs[2].norm, 1.197236));

    cout << "\n\n\n[PASSED] Da vuot qua tat ca cac testcase chang 1 & 2!!!!!\n\n\n";
}

void testPhase3(const SearchEngine& engine)
{
    const vector<Document>& docs = engine.getDocument();
    QueryVector qv = engine.vectorizeQuery("machine learning");
    assert(qv.unknownWords.empty());
    assert(isClose(qv.weights.at("machine"), 0.643841036));
    assert(isClose(qv.weights.at("learning"), 0.846573590));
    assert(isClose(qv.norm, 1.063587));

    QueryVector qKnown = engine.vectorizeQuery("machine");
    QueryVector qMixed = engine.vectorizeQuery("machine unknownword");
    assert(qMixed.unknownWords.size() == 1 && qMixed.unknownWords[0] == "unknownword");
    assert(isClose(qKnown.weights.at("machine"), qMixed.weights.at("machine")));
    assert(isClose(qKnown.norm, qMixed.norm));

    double selfSim = SearchEngine::computeCosine(docs[0].weights, docs[0].norm, docs[0].weights, docs[0].norm);
    assert(isClose(selfSim, 1));
    double zeroSim = SearchEngine::computeCosine(docs[0].weights, docs[0].norm, docs[2].weights, docs[2].norm);
    assert(isClose(zeroSim, 0));

    SearchResponse res = engine.searchBaseline("machine learning", 5);
    assert(res.totalMatched == 2);
    assert(res.results.size() == 2);
    assert(res.results[0].documentId == 0);
    assert(isClose(res.results[0].score, 0.943085997));
    assert(res.results[1].documentId == 1);
    assert(isClose(res.results[1].score, 0.366446816));

    SearchResponse top1 = engine.searchBaseline("machine learning", 1);
    assert(top1.totalMatched == 2 && top1.results.size() == 1 && top1.results[0].documentId == 0);
    SearchResponse top100 = engine.searchBaseline("machine learning", 100);
    assert(top100.results.size() == 2);

    cout << "\n\n\n[PASSED] Kiem thu Chang 3 (Vector Query, Cosine, Ranking Top K va Test bien )!!!!\n\n\n";
}

void printSearchResponse(const SearchEngine& engine, const string& query, int k)
{
    cout << "----------------------------------\n";
    cout << "Query: \"" << query << "\" (Top K = " << k << ")\n";
    if (k <= 0)
    {
        cout << "\n[N] K phai la so nguyen duong\n";
        return;
    }

    SearchResponse res = engine.searchBaseline(query, k);

    if (!res.queryVec.unknownWords.empty())
    {
        cout << "\n[W] Bo qua cac tu :\n";
        for (const string& word: res.queryVec.unknownWords)
            cout << "'" << word << "'";
        cout << "\n";
    }
    if (res.queryVec.isEmptyAfterPreprocess)
    {
        cout << "\n[R] Query rong, chi chua toan cac stop words!!!";
        return;
    }
    if (res.queryVec.weights.empty())
    {
        cout << "\n[R] Khong co tu khoa trong bo du lieu!!!";
        return;
    }
    if (res.results.empty())
    {
        cout << "\n[R] Khong tim thay tai lieu phu hop!!!";
        return;
    }
    const vector<Document>& docs = engine.getDocument();
    cout << "\nTim thay tong cong " << res.totalMatched << "!!!";
    cout << "\nTop " << k << ": \n";
    cout << fixed << setprecision(6);
    for (size_t i = 0; i < res.results.size(); i++)
    {
        int docId = res.results[i].documentId;
        double score = res.results[i].score;
        cout << " #" << i + 1 << " | ID: " << docId << " | File name: " << docs[docId].filename 
             << "| Score: " << score << "\n";
    }
}