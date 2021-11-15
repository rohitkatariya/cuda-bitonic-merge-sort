#include <cassert>
#include <random>
#include<iostream>
#include<stdio.h>
#include <fstream>
#include "sortcu.h"
using namespace std;

int main(int argc, char *argv[]) {
    int ndata;
    // ndata = stoi(argv[1]);
    // ndata = pow(2,ndata);
    // printf("ndata %d,RAND_MAX.:%ld",ndata,RAND_MAX);
    cin>>ndata;
    uint32_t *data = new uint32_t[ndata];
    for(int i=0;i<ndata;i++){
        cin>>data[i];
    }
    // cout<<"\n";
    // for(int i=0;i<ndata;i++){
    //     cout<<data[i]<<"\t";
    // }
    sort(data, ndata);
    // cout<<"\nsorted:";
    // for(int i=0;i<ndata;i++){
    //     cout<<data[i]<<" ";
    // }
}
