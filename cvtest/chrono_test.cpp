#include <iostream>
#include <chrono>
#include <thread>

using namespace std;
using namespace std::chrono;

void system_clock_test()
{
	auto start = system_clock::now();
	cout <<"start"<< endl;
	cout << start.time_since_epoch().count() << endl;
	this_thread::sleep_for(milliseconds(1000));

	auto stop = system_clock::now();
	cout << "stop" << endl;
	cout << stop.time_since_epoch().count() << endl;
	cout << stop.time_since_epoch().count() - start.time_since_epoch().count() << endl;

	auto elapsed = stop - start;

	long long miliseconds = std::chrono::duration_cast<milliseconds>(elapsed).count();

	cout << miliseconds << " ms" << endl;
}

void steady_clock_test()
{
	auto start = steady_clock::now();
	cout << "start" << endl;

	this_thread::sleep_for(milliseconds(1000));

	auto stop = steady_clock::now();
	cout << "stop" << endl;

	auto  elapsed = stop - start;

	long long miliseconds = duration_cast<milliseconds>(elapsed).count();

	cout << miliseconds << " ms" << endl;
}

int main(int argc, char* argv[])
{
	system_clock_test();

	steady_clock_test();

	return 0;
}