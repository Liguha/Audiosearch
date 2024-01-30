#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#define MINIMP3_IMPLEMENTATION
#include "minimp3.h"
#include "minimp3_ex.h"

using namespace std;
using cmd = complex <double>;

const double pi = acos(-1);
const int N = 4096, K = 2048;

struct vec6
{
    short v[6];

    vec6()
    {
        for (int i = 0; i < 6; i++)
            v[i] = 0;
    }
};

struct fingerprint
{
    string name = "file";
    vector <vec6> v;

    fingerprint() {};
    fingerprint(vector <vec6>& _v)
    {
        v = _v;
    }

    vec6& operator[](int idx)
    {
        return v[idx];
    }

    int size()
    {
        return v.size();
    }

    void serialize(ofstream& fout)
    {
        int n = name.length();
        fout.write((char*)&n, sizeof(int));
        fout.write(name.c_str(), n);
        n = v.size();
        fout.write((char*)&n, sizeof(int));
        for (int i = 0; i < n; i++)
            fout.write((char*)&v[i], sizeof(vec6));
    }

    static fingerprint deserialize(ifstream& fin)
    {
        fingerprint res;
        int n;
        fin.read((char*)&n, sizeof(int));
        res.name.resize(n);
        fin.read((char*)res.name.c_str(), n);
        fin.read((char*)&n, sizeof(int));
        for (int i = 0; i < n; i++)
        {
            vec6 cur;
            fin.read((char*)&cur, sizeof(vec6));
            res.v.push_back(cur);
        }
        return res;
    }
};

double operator-(const vec6& lhs, const vec6& rhs)
{
    int k = 0;
    for (int i = 0; i < 6; i++)
    {
        if (abs(lhs.v[i] - rhs.v[i]) < 3)
            k++;
    }
    return (k >= 3) ? 0 : 2;
}

vector <short> decode(string name)
{
    mp3dec_t mp3d;
    mp3dec_file_info_t info;
    if (mp3dec_load(&mp3d, name.c_str(), &info, NULL, NULL))
    {
        throw std::runtime_error("Decode error");
    }
    vector <short> file(info.buffer, info.buffer + info.samples);
    free(info.buffer);
    vector <short> res;
    for (int i = 0; i < file.size(); i += info.channels)
    {
        int sum = 0;
        for (int j = 0; j < info.channels; j++)
            sum += file[i + j];
        res.push_back(sum / info.channels);
    }
    return res;
}

vector <cmd> tmp(N);
void fft_r(vector <double>& p, vector <cmd>& res, cmd w, int l, int r)
{
    if (l == r)
    {
        int k = p.size();
        int n = 0;
        while (k > 1)
        {
            n++;
            k /= 2;
        }
        int c = 0;
        for (int i = 0; i < n; i++)
        {
            c = c | (l & 1);
            c = c << 1;
            l = l >> 1;
        }
        c = c >> 1;
        res[r] = p[c];
        return;
    }
    int m = (l + r) / 2;
    fft_r(p, res, w * w, l, m);
    fft_r(p, res, w * w, m + 1, r);
    int n = r - l + 1;
    int k = n / 2;
    cmd wt = 1.0;
    for (int i = 0; i < n; i++)
    {
        cmd a = res[l + i % k] + wt * res[m + 1 + i % k];
        tmp[l + i] = a;
        wt *= w;
    }
    for (int i = 0; i < n; i++)
        res[l + i] = tmp[l + i];
}

void fft(vector <double>& v, vector <cmd>& out)
{
    int n = v.size();
    fft_r(v, out, polar((double)1.0, 2 * pi / n), 0, n - 1);
}

vector <int> lim = { 0, 40, 80, 160, 320, 640, 2048 };
vector <vec6> get_vec6(vector <vector <cmd>>& spectr)
{
    vector <vec6> res;
    for (int i = 0; i < spectr.size(); i++)
    {
        vec6 cur;
        vector <double> mx(6);
        for (int l = 0; l < 6; l++)
        {
            for (int j = lim[l]; j < lim[l + 1]; j++)
            {
                double lvl = abs(spectr[i][j]);
                if (lvl > mx[l])
                {
                    mx[l] = lvl;
                    cur.v[l] = j;
                }
            }
        }
        res.push_back(cur);
    }
    return res;
}

fingerprint get_fingerpint(vector <short>& samples)
{
    vector <double> window(N);
    vector <cmd> fft_window(N);
    vector <double> cosv(N);
    for (int i = 0; i < N; i++)
        cosv[i] = cos(2 * pi * i / (N - 1));

    vector <vector <cmd>> spectr;
    for (int i = 0; i < samples.size() - N; i += K)
    {
        for (int j = 0; j < N; j++)
        {
            window[j] = samples[i + j] * 0.5 * (1 - cosv[j]);
        }
        fft(window, fft_window);
        spectr.push_back(fft_window);
    }
    vector <vec6> v = get_vec6(spectr);
    return fingerprint(v);
}

double similarity(int l, fingerprint& t, fingerprint& p)
{
    int n = min(t.size(), p.size());
    vector <vector <short>> dp(n + 1, vector <short>(n + 1));
    for (int i = 1; i <= n; i++)
    {
        dp[0][i] = i;
        dp[i][0] = i;
    }
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            int a = dp[i][j - 1] + 1;
            int b = dp[i - 1][j] + 1;
            int c = dp[i - 1][j - 1] + (t[l + i - 1] - p[j - 1]);
            dp[i][j] = min(min(a, b), c);
        }
    }
    return 1 - dp[n][n] / (2.0 * n);
}

double max_similarity(fingerprint& full, fingerprint& part)
{
    int n = part.size();
    int m = full.size();
    double prev = similarity(0, full, part);
    double mx = 0;
    for (int i = n; i < m - n; i = min(i + n, m - n))
    {
        double cur = similarity(i, full, part);
        int l = i - n, r = i + 1;
        double ls = prev, rs = cur;
        if (prev + cur < 0.4)
        {
            prev = cur;
            continue;
        }
        for (int j = 0; j < 10; j++)
        {
            int m = (l + r) / 2;
            double ms = similarity(m, full, part);
            mx = max(mx, ms);
            if (ls > rs)
            {
                r = m;
                rs = ms;
            }
            else
            {
                l = m;
                ls = ms;
            }
        }
        prev = cur;
    }
    return mx;
}

int main(int argc, char* argv[])
{
    string action;
    ifstream idx;
    ifstream fin;
    ofstream fout;
    action = argv[1];
    for (int i = 2; i < argc; i++)
    {
        if (string(argv[i]) == "--index")
            idx = ifstream(argv[i + 1], ios::binary);
        if (string(argv[i]) == "--input")
            fin = ifstream(argv[i + 1]);
        if (string(argv[i]) == "--output")
            fout = ofstream(argv[i + 1], ios::binary);
    }
    vector <string> input;
    while (!fin.eof())
    {
        string name;
        getline(fin, name);
        input.push_back(name);
    }
    if (action == "index")
    {
        int n = input.size();
        fout.write((char*)&n, sizeof(int));
        for (int i = 0; i < n; i++)
        {
            vector <short> samples = decode(input[i]);
            fingerprint f = get_fingerpint(samples);
            f.name = input[i];
            f.serialize(fout);
        }
    }
    if (action == "search")
    {
        int n;
        idx.read((char*)&n, sizeof(int));
        vector <fingerprint> v(n);
        for (int i = 0; i < n; i++)
            v[i] = fingerprint::deserialize(idx);
        int m = input.size();
        for (int i = 0; i < m; i++)
        {
            vector <short> samples = decode(input[i]);
            fingerprint f = get_fingerpint(samples);
            int ans = -1;
            double mx = 0.4;
            for (int j = 0; j < n; j++)
            {
                double cur = max_similarity(v[j], f);
                if (cur > mx)
                {
                    mx = cur;
                    ans = j;
                }
            }
            cout << mx << '\n';
            if (ans != -1)
            {
                fout << v[ans].name << '\n';
            }
            else
                fout << "! NOT FOUND\n";
        }
        idx.close();
    }
    fin.close();
    fout.close();
}
