#include <cassert>
#include <random>
#include<iostream>
#include<stdio.h>
#include <fstream>
// #include "sortcu.h"
using namespace std;

int main(int argc, char *argv[]) {
    int ndata;
    ndata = stoi(argv[1]);
    srand(time(0));
    // ndata = pow(2,ndata);
    // printf("ndata %d,RAND_MAX.:%ld",ndata,RAND_MAX);
    cout<<ndata;
    uint32_t *data = new uint32_t[ndata];
    for(int i=0;i<ndata;i++){
        data[i]=rand();
    }
    // cout<<"\n";
    // for(int i=0;i<ndata;i++){
    //     cout<<data[i]<<"\t";
    // }
    // sort(data, ndata);
    cout<<"\n";
    for(int i=0;i<ndata;i++){
        cout<<data[i]<<" ";
    }
}
