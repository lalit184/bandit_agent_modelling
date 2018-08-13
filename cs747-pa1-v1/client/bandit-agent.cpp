#include <iostream>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <vector>
#include <random>
#include <string>
#include <time.h>
#include <boost/math/distributions/beta.hpp>
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

#define MAXHOSTNAME 256

using namespace std;
using namespace boost::math;


void options(){

  cout << "Usage:\n";
  cout << "bandit-agent\n"; 
  cout << "\t[--numArms numArms]\n";
  cout << "\t[--randomSeed randomSeed]\n";
  cout << "\t[--horizon horizon]\n";
  cout << "\t[--hostname hostname]\n";
  cout << "\t[--port port]\n";
  cout << "\t[--algorithm algorithm]\n";
  cout << "\t[--epsilon epsilon]\n";

}


/*
  Read command line arguments, and set the ones that are passed (the others remain default.)
*/
bool setRunParameters(int argc, char *argv[], int &numArms, int &randomSeed, unsigned long int &horizon, string &hostname, int &port, string &algorithm, double &epsilon){

  int ctr = 1;
  while(ctr < argc){

    //cout << string(argv[ctr]) << "\n";

    if(string(argv[ctr]) == "--help"){
      return false;//This should print options and exit.
    }
    else if(string(argv[ctr]) == "--numArms"){
      if(ctr == (argc - 1)){
	return false;
      }
      numArms = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--randomSeed"){
      if(ctr == (argc - 1)){
	return false;
      }
      randomSeed = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--horizon"){
      if(ctr == (argc - 1)){
	return false;
      }
      horizon = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--hostname"){
      if(ctr == (argc - 1)){
	return false;
      }
      hostname = string(argv[ctr + 1]);
      ctr++;
    }
    else if(string(argv[ctr]) == "--port"){
      if(ctr == (argc - 1)){
	return false;
      }
      port = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--algorithm"){
      if(ctr == (argc - 1)){
  return false;
      }
      algorithm = string(argv[ctr + 1]);
      ctr++;
    }
     else if(string(argv[ctr]) == "--epsilon"){
      if(ctr == (argc - 1)){
  return false;
      }
      epsilon = atof(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else{
      return false;
    }

    ctr++;
  }

  return true;
}

/* ============================================================================= */
/* Write your algorithms here */
/*
int sampleArm(string algorithm, double epsilon, int pulls, float reward, int numArms,epsilon_greedy eppo){
  if(algorithm.compare("rr") == 0){
    return(pulls % numArms);
  }
  else if(algorithm.compare("epsilon-greedy") == 0){
    return(eppo.update(reward));
  }
  else if(algorithm.compare("UCB") == 0){
    return(pulls % numArms);
  }
  else if(algorithm.compare("KL-UCB") == 0){
    return(pulls % numArms);
  }
  else if(algorithm.compare("Thompson-Sampling") == 0){
    return(pulls % numArms);
  }
  else{
    return -1;
  }
}
*/
/* ============================================================================= */

class epsilon_greedy
{
	public:
		int arm=-1;
		int n_arms;
		double epsilon;
		float* average_reward;
		float* pulls;
		
		epsilon_greedy(int num_arms,double eps)
		{	
			
			n_arms=num_arms;
			average_reward = new float[num_arms];
			pulls =new float[num_arms];
      for(int i=0;i<n_arms;i++)
      {
        pulls[i]=0;
        average_reward[i]=0;
      }
			epsilon=eps;
		}	
		
	int update(float reward,int pull)
	{
		if(arm >=0) {
			average_reward[arm] = ((average_reward[arm] * pulls[arm]) + reward)/(pulls[arm] + 1);
			pulls[arm]+=1;
		
  		if(rand()%1000 < (1000*epsilon)) {
        arm= rand()%n_arms;
      }

      else {
        float max=0;
        
        for(int i=0;i<n_arms;i++) {
          float avg_r;
          if(pulls[i]>0) {
            avg_r=float(average_reward[i]);
            //cout<<"arm " <<i<<" has avg "<<cumulative_reward[i];
          }
          
          else
            avg_r=0;

          if(max < avg_r) {
            max=avg_r;
            arm=i;
          }
          cout<<endl;
        }
      }
      //cout << "arm="<<arm<<endl;
      return arm; 
    }
    else {
      arm=rand()%n_arms;
      //cout << "First run, arm="<<arm<<endl;
      return arm;
    }
	}
};

class UCB
{
  public:
    int arm=-1;
    int n_arms;
    float* average_reward;
    float* pulls;
    int init_flag=0;
    
    UCB(int num_arms)
    { 
      
      n_arms=num_arms;
      average_reward = new float[num_arms];
      pulls =new float[num_arms];
      for(int i=0;i<n_arms;i++)
      {
        pulls[i]=0;
        average_reward[i]=0;
      }
    } 
    
  int update(float reward,int pull)
  {
    if(init_flag!=0) {
      average_reward[arm] = ((average_reward[arm] * pulls[arm]) + reward)/(pulls[arm] + 1);
      pulls[arm]+=1;
  
      float max=0;
      
      for(int i=0;i<n_arms;i++) {
        float ucb;
      
        ucb = average_reward[i] +sqrt(2*log10(pull)/pulls[i]);
        
        if(max < ucb) {
          max=ucb;
          arm=i;
        }
        cout<<endl;
      }
    
      //cout << "arm="<<arm<<endl;
      return arm; 
    }
    else {
      arm=pull%n_arms;
      if(pull == n_arms-1)
        init_flag=1;
      //cout << "First run, arm="<<arm<<endl;
      return arm;
    }
  }
};

class Thompson{
  public:
    float*success;
    float* failures;
    int arm=-1;
    int n_arms;
    

    /*
    template<typename URNG>
    URNG engine;
    double beta_sample(URNG& engine,double a,double b){
      uniform_real_distribution<double> unif(0,1);
      double p=unif(engine);
      return boost::math::ibeta(a,b,p);
    }
    */
    Thompson(int num_arms)
    {
      n_arms=num_arms;
      success = new float[n_arms];
      failures = new float[n_arms];
      
    }

    int update(int reward){
      if(arm<0){
        
      }
      else{
        if(reward==0)
          failures[arm]+=1;
        
        else
          success[arm]+=1;
      }
      
      arm =sample();
      return arm;
    }

    int sample(){
      float max_tomsample=0;
      for(int i=0;i<n_arms;i++){
        float uniform =float((rand()%1000)/1000.0);
        beta_distribution<> dist(success[i]+1,failures[i]+1);
        double samp = quantile(dist,uniform);
        if(max_tomsample<samp){
          arm=i;
          max_tomsample=samp;
        }
      }
     return arm; 
    }
};


class KL_UCB
{
  public:
    int arm=-1;
    int n_arms;
    float* average_reward;
    float* pulls;
    int init_flag=0;
    
    KL_UCB(int num_arms)
    { 
      
      n_arms=num_arms;
      average_reward = new float[num_arms];
      pulls =new float[num_arms];
      for(int i=0;i<n_arms;i++)
      {
        pulls[i]=0;
        average_reward[i]=0;
      }
    } 
    
  int update(float reward,int pull)
  {
    if(init_flag!=0) {
      average_reward[arm] = ((average_reward[arm] * pulls[arm]) + reward)/(pulls[arm] + 1);
      pulls[arm]+=1;
  
      float max=0;
      
      for(int i=0;i<n_arms;i++) {
        float ucb;
      
        ucb = average_reward[i] +(  log10(pull) + 3*log10( log10(pull) )  )/pulls[i] ;
        
        if(max < ucb) {
          max=ucb;
          arm=i;
        }
        cout<<endl;
      }
    
      //cout << "arm="<<arm<<endl;
      return arm; 
    }
    else {
      arm=pull%n_arms;
      if(pull == n_arms-1)
        init_flag=1;
      //cout << "First run, arm="<<arm<<endl;
      return arm;
    }
  }
};


int main(int argc, char *argv[]){
  // Run Parameter defaults.
  int numArms = 5;
  srand(time(0));
  int randomSeed = time(0);
  unsigned long int horizon = 200;
  string hostname = "localhost";
  int port = 5000;
  string algorithm="random";
  double epsilon=0.0;

  //Set from command line, if any.
  if(!(setRunParameters(argc, argv, numArms, randomSeed, horizon, hostname, port, algorithm, epsilon))){
    //Error parsing command line.
    options();
    return 1;
  }
  epsilon_greedy eppo(numArms,epsilon);
  UCB adversity(numArms);
  Thompson tom(numArms);
  KL_UCB kl_ucb(numArms);
  struct sockaddr_in remoteSocketInfo;
  struct hostent *hPtr;
  int socketHandle;

  bzero(&remoteSocketInfo, sizeof(sockaddr_in));
  
  if((hPtr = gethostbyname((char*)(hostname.c_str()))) == NULL){
    cerr << "System DNS name resolution not configured properly." << "\n";
    cerr << "Error number: " << ECONNREFUSED << "\n";
    exit(EXIT_FAILURE);
  }

  if((socketHandle = socket(AF_INET, SOCK_STREAM, 0)) < 0){
    close(socketHandle);
    exit(EXIT_FAILURE);
  }

  memcpy((char *)&remoteSocketInfo.sin_addr, hPtr->h_addr, hPtr->h_length);
  remoteSocketInfo.sin_family = AF_INET;
  remoteSocketInfo.sin_port = htons((u_short)port);

  if(connect(socketHandle, (struct sockaddr *)&remoteSocketInfo, sizeof(sockaddr_in)) < 0){
    //code added
    cout<<"connection problem"<<".\n";
    close(socketHandle);
    exit(EXIT_FAILURE);
  }


  char sendBuf[256];
  char recvBuf[256];

  float reward = 0;
  unsigned long int pulls=0;
  int armToPull = kl_ucb.update(reward,pulls);//sampleArm(algorithm, epsilon, pulls, reward, numArms,eppo);
  
  sprintf(sendBuf, "%d", armToPull);

  cout << "Sending action " << armToPull << ".\n";
  while(send(socketHandle, sendBuf, strlen(sendBuf)+1, MSG_NOSIGNAL) >= 0){

    char temp;
    recv(socketHandle, recvBuf, 256, 0);
    sscanf(recvBuf, "%f %c %lu", &reward, &temp, &pulls);
    cout << "Received reward " << reward << ".\n";
    cout<<"Num of  pulls "<<pulls<<".\n";


    armToPull = kl_ucb.update(reward,pulls);//sampleArm(algorithm, epsilon, pulls, reward, numArms,eppo);

    sprintf(sendBuf, "%d", armToPull);
    cout << "Sending action " << armToPull << ".\n";
  }
  
  close(socketHandle);

  cout << "Terminating.\n";

  return 0;
}
          
