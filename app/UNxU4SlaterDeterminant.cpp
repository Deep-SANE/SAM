#include <iostream>

#include <qcx_utils.h>

#include <UNxU4SlaterDeterminant.h>

int main(int argc, char* argv[])
{
  qcx_utils::initialize(&argc, &argv);

  int n;
  std::cout << "Enter H.O. shell: ";
  std::cin >> n;

  UNxU4::SlaterDeterminant sd(n);

  std::cout << "Where to occupy? Enter two indices: (-1 to stop)" << std::endl;
  int r, c;
  while (true)
  {
    std::cin >> r >> c;
    if (r < 0 || c < 0)
      break;
    sd.occupy(r, c);
  }
  std::cout << "Your SD is \n" << sd << std::endl;
  // sd.print();

  bool decision;
  std::cout << "Want to vacate any state? 1 for yes, 0 for no ";
  std::cin >> decision;

  if (decision == true)
  {
    int i, j;
    std::cout << "Where to vacate? Enter two indices: (-1 to stop)"
              << std::endl;
    while (true)
    {
      std::cin >> i >> j;
      if (i < 0 || j < 0)
        break;
      sd.vacate(i, j);
    }
    std::cout << "Your SD has changed to \n" << sd << std::endl;
    // sd.print();
  }

  std::cout << "Count preceding occupation: ";
  std::cin >> r >> c;
  std::cout << sd.preceding_occupation(r, c);

  qcx_utils::finalize();
}
