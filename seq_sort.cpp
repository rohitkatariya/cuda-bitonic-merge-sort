#include<bits/stdc++.h>
#include <string>
using namespace std;


    bool compare(const pair<uint32_t, uint32_t>&i, const pair<uint32_t, uint32_t>&j)
    {
        return i.second < j.second;
    }

int main(){

int n;// = 1073741824;//536870912;//268435456;//134217728;//67108864;//1073741824;//536870912;
   cin>>n;
   ofstream of;
   char filename[80];
  strcpy (filename,"input_dir/seqout_");
  strcat (filename,to_string(n).c_str());
  strcat (filename,".txt");
  of.open(filename);
  cout<<filename<<"\n";
   pair<uint32_t,uint32_t> *data = new pair<uint32_t,uint32_t>[n];
   for(int i=0;i<n;i++){
       uint32_t tmp;
       cin>>tmp;
       data[i]=make_pair(tmp,0);
   }
   data[0].second = data[0].first;
   for(int i=1;i<n;i++){
        data[i].second = (long(data[i-1].second)+long(data[i].first))%4294967295;
  }
    sort(data,data+n,compare);

    for(int i=0;i<n;i++)
    {
        if(i>0 && data[i].second==data[i-1].second){
            continue;
        }
        if(i<n-1  && data[i].second==data[i+1].second){
            continue;
        }
        of<<data[i].first<<"_"<<data[i].second<<" ";
        if(i%10==0){
            of<<"\n";
        }
    }
}
