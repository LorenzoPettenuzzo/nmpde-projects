#include "LinearFisherKolmogorov.hpp"
#include "FisherKolmogorov.hpp"

int main(int argc, char *argv[]) {

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    //LinearFisherKolmogorov<3> lfk("../mesh/brain-h3.0.msh", 1, 10, 0.01);
    //LinearFisherKolmogorov<3> lfk("../mesh/brain-h3.0.msh", 1, 4, 0.25);
    FisherKolmogorov<3> lfk("../mesh/brain-h3.0.msh", 1, 40, 1.0);
    lfk.setup();
    lfk.solve();
}