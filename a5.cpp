#include <cassert>
#include <random>
#include<iostream>
#include<stdio.h>
#include <fstream>
#include "sortcu.h"
using namespace std;

int main(int argc, char *argv[]) {
    int ndata;
    ndata = stoi(argv[1]);
    printf("ndata %d,",ndata);

    uint32_t *data = new uint32_t[ndata];
    for(int i=0;i<ndata;i++){
        data[i]=rand()%1000;
    }
    sort(data, ndata);
    // for(int i=0;i<ndata;i++){
    //     cout<<data[i]<<"\t";
    // }
}
