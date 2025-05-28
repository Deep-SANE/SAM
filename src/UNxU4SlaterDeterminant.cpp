#include <iostream>

#include "../include/UNxU4SlaterDeterminant.h"

int main()
{
  UNxU4::SlaterDeterminant sd(2);
  sd.occupy(0, 1);
  sd.occupy(2, 2);
  sd.occupy(1, 5);
  std::cout << "My SD is" << std::endl;
  sd.print();

  std::cout << "Nucleons: " << sd.occupation() << std::endl;

  std::cout << "State (0,1) is occupied? "
            << (sd.is_occupied(0, 1) ? "Yes" : "No") << std::endl;

  sd.vacate(0, 1);
  std::cout << "My SD has changed to" << std::endl;
  sd.print();

  std::cout << "Hopping over " << sd.num_hop(2, 3, 0, 0) << " nucleons"
            << std::endl;

  std::cout << "State (0,1) is occupied? "
            << (sd.is_occupied(0, 1) ? "Yes" : "No") << std::endl;

  return 0;
}
