#include <cassert>
#include <random>
#include<iostream>
#include<stdio.h>
#include<bits/stdc++.h>
#include <string>
#include "sortcu.h"
using namespace std;

int main(int argc, char *argv[]) {
    int ndata;
    ndata = stoi(argv[1]);
    // ndata = stoi(argv[1]);
    // ndata = pow(2,ndata);
    // printf("ndata %d,RAND_MAX.:%ld",ndata,RAND_MAX);
    ifstream fin;
    char filename[80];
  strcpy (filename,"input_dir/input_");
  strcat (filename,to_string(ndata).c_str());
  strcat (filename,".txt");
  fin.open(filename);
    
    fin>>ndata;
    uint32_t *data = new uint32_t[ndata];


    for(int i=0;i<ndata;i++){
        fin>>data[i];
    }
    cout<<filename<<"\n";
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
