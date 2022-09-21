#include "mylib.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <map>
#include <stdlib.h>
#include <set>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include "graph.h"
#include "config.h"
#include "algo.h"
#include "query.h"
#include "build.h"
#include "omp.h"

#include <boost/progress.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace std::chrono;

using namespace boost;
using namespace boost::property_tree;

using namespace std;

class Fora_class{
    
public:
    protected:
    unordered_map<int, PredResult> pred_results;
    Fwdidx fwd_idx;
    IFwdidx ifwd_idx;
    Bwdidx bwd_idx;
    //iMap<double> ppr;
    vector<double> ppr;
    iMap<int> topk_filter;

private:
    Graph &graph;
    int worker_num;
    std::minstd_rand rd;

//-----------------voids------------------------------------

public:

    Fora_class(Graph &_graph, int _worker_num): graph(_graph), worker_num(_worker_num)
    {
        init();
    }

    void init(){
        ppr.resize(graph.n+1);
        fwd_idx.first.nil = -1;
        fwd_idx.second.nil =-1;
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        rd.seed(worker_num+1);
    }

    void forward_local_update_linear_CLASS(int s, /*const Graph &graph,*/ double& rsum, double rmax, double init_residual = 1.0){
        fwd_idx.first.clean();
        fwd_idx.second.clean();

        static vector<bool> idx(graph.n);
        std::fill(idx.begin(), idx.end(), false);

        if(graph.g[s].size()==0){
            fwd_idx.first.insert( s, 1);
            rsum =0;
            return; 
        }

        double myeps = rmax;//config.rmax;

        vector<int> q;  //nodes that can still propagate forward
        q.reserve(graph.n);
        q.push_back(-1);
        unsigned long left = 1;
        q.push_back(s);

    // residual[s] = init_residual;
        fwd_idx.second.insert(s, init_residual);
    
        idx[s] = true;
    
        while (left < (int) q.size()) {
            int v = q[left];
            idx[v] = false;
            left++;
            double v_residue = fwd_idx.second[v];
            fwd_idx.second[v] = 0;
            if(!fwd_idx.first.exist(v))
                fwd_idx.first.insert( v, v_residue * config.alpha);
            else
                fwd_idx.first[v] += v_residue * config.alpha;

            int out_neighbor = graph.g[v].size();
            rsum -=v_residue*config.alpha;
            if(out_neighbor == 0){
                fwd_idx.second[s] += v_residue * (1-config.alpha);
                if(graph.g[s].size()>0 && fwd_idx.second[s]/graph.g[s].size() >= myeps && idx[s] != true){
                    idx[s] = true;
                    q.push_back(s);   
                }
                continue;
            }

            double avg_push_residual = ((1.0 - config.alpha) * v_residue) / out_neighbor;
            for (int next : graph.g[v]) {
                // total_push++;
                if( !fwd_idx.second.exist(next) )
                    fwd_idx.second.insert( next,  avg_push_residual);
                else
                    fwd_idx.second[next] += avg_push_residual;

                //if a node's' current residual is small, but next time it got a laerge residual, it can still be added into forward list
                //so this is correct
                if ( fwd_idx.second[next]/graph.g[next].size() >= myeps && idx[next] != true) {  
                    idx[next] = true;//(int) q.size();
                    q.push_back(next);    
                }
            }
        }
    }

    inline int random_walk_CLASS(int start/*, const Graph& graph*/, int& check_path_length){
        int cur = start;
        unsigned long k;
        unsigned int seed=worker_num;
        if(graph.g[start].size()==0){
            return start;
        }
        while (true) {
            int temp=rd();
            //printf("check random: %d\n", temp);
            //printf("check threshold: %d\n", config.alpha*std::minstd_rand::max());
            if ((temp)<(config.alpha*std::minstd_rand::max())) {
                return cur;
            }
            if (graph.g[cur].size()){
                //k = graph.g[cur].size() * (rd())*(1.0 / (std::minstd_rand::max() + 1.0));
                k = rd() % graph.g[cur].size();
                cur = graph.g[cur][ k ];
                check_path_length++;
            }
            else{
                cur = start;
            }
        }
    }

    void compute_ppr_with_fwdidx_CLASS(/*const Graph& graph,*/ double check_rsum){
        //ppr.reset_zero_values();
        std::fill(ppr.begin(), ppr.end(), 0);
        int node_id;
        double reserve;
        for(long i=0; i< fwd_idx.first.occur.m_num; i++){
            node_id = fwd_idx.first.occur[i];
            reserve = fwd_idx.first[ node_id ];
            ppr[node_id] = reserve;
        }
        // INFO("rsum is:", check_rsum);
        if(check_rsum == 0.0)
            return;

        unsigned long long num_random_walk = config.omega*check_rsum;
        // INFO(num_random_walk);
        //num_total_rw += num_random_walk;
        {
            Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk
            //Timer tm(SOURCE_DIST);
            double OMP_check_walk_start=omp_get_wtime();
            //----------------Random Walks without Index---------------------------------------------------------
               int check_num_walks=0;
               int check_path_walks=0;
               //printf("Check Num of nodes: %d\n", fwd_idx.second.occur.m_num);
               for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                   int source = fwd_idx.second.occur[i];
                   double residual = fwd_idx.second[source];
                   unsigned long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                   double a_s = residual/check_rsum*num_random_walk/num_s_rw;
                   double ppr_incre = a_s*check_rsum/num_random_walk;
                   num_total_rw += num_s_rw;
                   check_num_walks+=num_s_rw;
                   for(unsigned long j=0; j<num_s_rw; j++){
                       int des = random_walk_CLASS(source/*, graph*/, check_path_walks);
                       ppr[des] += ppr_incre;
                   }
               }
               //printf("------------\n");
               //printf("Check alpha is: %.6f\n", config.alpha);
               //printf("Check Num Walks is: %d\n", check_num_walks);
               //printf("Check length of walks is: %d\n", check_path_walks);
               //printf("Check average path length of walks is: %.6f\n", ((double)(check_path_walks))/((double)(check_num_walks)));
               //printf("------------\n");
            //----------------Random Walks without Index---------------------------------------------------------
            double OMP_check_walk_end=omp_get_wtime();
            //printf("Check time of Walks is: %.12f\n", OMP_check_walk_end-OMP_check_walk_start);
            //printf("------------\n");
        }
    }

    double fora_class_query_basic_CLASS(int v, double& push_time, double& walk_time){
        double rsum = 1.0;
        double total_rsum=0;
        int num_walk=0;
        //printf("point1: thread number %d\n", omp_get_thread_num());
        double OMP_check_time_start=omp_get_wtime();
        forward_local_update_linear_CLASS(v, /*graph,*/ rsum, config.rmax);
        double OMP_check_time_middle=omp_get_wtime();
        push_time=OMP_check_time_middle-OMP_check_time_start;
        //printf("Thread: %d, Check time of push is: %.12f\n", worker_num, OMP_check_time_middle-OMP_check_time_start);
        //printf("check RSUM: %.6f\n", rsum);
        //printf("point2\n");

        
        compute_ppr_with_fwdidx_CLASS(/*graph,*/ rsum);
        double OMP_check_time_end=omp_get_wtime();
        walk_time=OMP_check_time_end-OMP_check_time_middle;
        //printf("Thread: %d, Check time of walk is: %.12f\n", worker_num, OMP_check_time_end-OMP_check_time_middle);
        //printf("CLASS: source: %d, check time: %.4f\n", v, OMP_check_time_end-OMP_check_time_start);
        //printf("point3\n");
        total_rsum+= rsum*(1-config.alpha);
        return total_rsum;
    }


};