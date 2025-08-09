#include "ImplicitSolver.hpp"

#include <chrono>

int main(int argc, char *argv[]) {

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    freopen("log.txt", "w", stdout);

    //LinearFisherKolmogorov<3> lfk("../mesh/brain-h3.0.msh", 1, 10, 0.01);
    //LinearFisherKolmogorov<3> lfk("../mesh/brain-h3.0.msh", 1, 4, 0.25);

    auto t_start = std::chrono::high_resolution_clock::now();

    FisherKolmogorov<3> lfk("../mesh/brain-h3.0.msh", 1, 40, 0.1);
    lfk.setup();
    lfk.solve();

    auto t_end = std::chrono::high_resolution_clock::now();

    auto t_m = std::chrono::duration_cast<std::chrono::minutes>(t_end - t_start);
    auto t_s = std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start);

    std::cout << t_m.count() << " minutes " << t_s.count() % 60 << " seconds" << std::endl;
}