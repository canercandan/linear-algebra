#include <cusp/csr_matrix.h>

using namespace cusp;

int main(void)
{
    // allocate storage for (4,3) matrix with 4 nonzeros
    csr_matrix<int,float,host_memory> A(4,3,6);

    // initialize matrix entries on host
    A.row_offsets[0] = 0;  // first offset is always zero
    A.row_offsets[1] = 2;
    A.row_offsets[2] = 2;
    A.row_offsets[3] = 3;
    A.row_offsets[4] = 6; // last offset is always num_entries

    A.column_indices[0] = 0; A.values[0] = 10;
    A.column_indices[1] = 2; A.values[1] = 20;
    A.column_indices[2] = 2; A.values[2] = 30;
    A.column_indices[3] = 0; A.values[3] = 40;
    A.column_indices[4] = 1; A.values[4] = 50;
    A.column_indices[5] = 2; A.values[5] = 60;

    // A now represents the following matrix
    //    [10  0 20]
    //    [ 0  0  0]
    //    [ 0  0 30]
    //    [40 50 60]

    // copy to the device
    csr_matrix<int,float,device_memory> B = A;

    print_matrix(B);

    return 0;
}
