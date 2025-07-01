#include <iostream>
#include <utility>

#include <U3basis.h>
#include <U4basis.h>

#include <U3xU4SingleShellTensors.h>

int main()
{
	int n = 2, num_ads = 2, num_ats = 2;

	std::vector<U3xU4::Tadat> tensors = 
		U3xU4::GenerateSingleShellTensors(n, num_ads, num_ats);

	for (auto t : tensors) t.print();

	return 0;
}