#include "MatrixMarket.hpp"

////////////////////////////////////////////////////////////////////////////////////////////
// real
////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void LoadMatrixMarket(char* file, int* n, int* total, int** rows, int** cols, T** coefs)
{
    FILE *fdata;
    int base = 0;
    bool header_specify_base=false;
    int nrows, ncols, nnz;
    const int BUF_SIZE = 256;
    char buf[BUF_SIZE];
    char float_type[32];
    char pattern_type[32];

    if ((fdata=fopen(file,"r")) == NULL)
	{
	    printf("Impossible d'ouvrir le fichier %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    fscanf(fdata,"%%%%MatrixMarket matrix coordinate %31s %31s\n",
	   float_type,pattern_type);
    if(0!=strcmp(float_type,"real"))
	{
	    printf("Complex number type in MatrixMarket file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    bool is_symmetric = (0==strcmp(pattern_type,"symmetric"));
    if(!is_symmetric && 0!=strcmp(pattern_type,"general"))
	{
	    printf("Format problem in MatrixMarket file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    while(1) {
	if(NULL==fgets(buf,BUF_SIZE,fdata))
	    {
		printf("Format problem in MatrixMarket file %s\n",file);
		exit(EXIT_FAILURE) ;
	    }
	if(buf[0]!='%') break; // should finish line... and trim the left...
	if(1==sscanf(buf,"%% base %i\n",&base)) header_specify_base=true;
    }
    if(3!=sscanf(buf,"%u %u %u",&nrows,&ncols,&nnz))
	{
	    printf("Format problem in MatrixMarket file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    if(nrows!=ncols){
	printf("Dimension problem in MatrixMarket file %s\n",file);
	exit(EXIT_FAILURE) ;
    }

    *n=nrows;
    *total=nnz;
    *rows=(int *)malloc((*total)*sizeof(int));
    *cols=(int *)malloc((*total)*sizeof(int));
    *coefs=(T *)malloc((*total)*sizeof(T));

    int row,col;
    double val;

    if(2!=fscanf(fdata,"%u %u",&row,&col))
	{
	    if(feof(fdata)!=0)
		{
		    printf("EOF encoutered in %s while reading entry\n", file);
		    return;
		}
	    else
		{
		    printf("Format problem in MatrixMarket file %s\n",file);
		    exit(EXIT_FAILURE) ;
		}
	}

    if(!header_specify_base && row==1)
	{
	    base=1;
	}
    else
	{
	    if(!header_specify_base && row!=0)
		{
		    printf("Base problem in MatrixMarket file %s\n",file);
		    exit(EXIT_FAILURE) ;
		}
	}
    if(1!=fscanf(fdata,"%lf",&val))
	{
	    printf("Format problem in MatrixMarket file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    int compt=0;
    (*rows)[compt]=row-base;
    (*cols)[compt]=col-base;
    (*coefs)[compt]=(T)val;
    compt++;
    if(is_symmetric && row!=col){
	(*rows)[compt]=col-base;
	(*cols)[compt]=row-base;
	(*coefs)[compt]=(T)val;
	compt++;
    }

    for(int i=1;i<nnz;++i) {
	if(2!=fscanf(fdata,"%u %u",&row,&col)) {
	    if(feof(fdata)!=0)
		{
		    printf("EOF encoutered in %s while reading entry\n", file);
		    return;
		}
	    else
		{
		    printf("Format problem in MatrixMarket file %s\n",file);
		    exit(EXIT_FAILURE) ;
		}
	}
	if(1!=fscanf(fdata,"%lf",&val))
	    {
		printf("Format problem in MatrixMarket file %s\n",file);
		exit(EXIT_FAILURE) ;
	    }
	(*rows)[compt]=row-base;
	(*cols)[compt]=col-base;
	(*coefs)[compt]=(T)val;
	compt++;
	if(is_symmetric && row!=col){
	    (*rows)[compt]=col-base;
	    (*cols)[compt]=row-base;
	    (*coefs)[compt]=(T)val;
	    compt++;
	}
    }
    fclose(fdata);
}
#define TEMPLATE_INSTANTIATION(T) \
void LoadMatrixMarket(char* file, int* n, int* total, int** rows, int** cols, T** coefs)
MM_INSTANCIATE_TEMPLATE(TEMPLATE_INSTANTIATION)
#undef TEMPLATE_INSTANTIATION
////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void LoadMatrixMarket_CSR(char* file, int* n, int* total, int** rows_index, int** cols, T** coefs)
{

    FILE *fdata;
    int base = 0;
    bool header_specify_base=false;
    int nrows, ncols, nnz;
    const int BUF_SIZE = 256;
    char buf[BUF_SIZE];
    char float_type[32];
    char pattern_type[32];

    if ((fdata=fopen(file,"r")) == NULL)
	{
	    printf("Impossible d'ouvrir le fichier %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    fscanf(fdata,"%%%%MatrixMarket matrix coordinate %31s %31s\n",
	   float_type,pattern_type);
    if(0!=strcmp(float_type,"real"))
	{
	    printf("Complex number type in MatrixMarket file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    bool is_symmetric = (0==strcmp(pattern_type,"symmetric"));
    if(!is_symmetric && 0!=strcmp(pattern_type,"general"))
	{
	    printf("Format problem in MatrixMarket file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    while(1) {
	if(NULL==fgets(buf,BUF_SIZE,fdata))
	    {
		printf("Format problem in MatrixMarket file %s\n",file);
		exit(EXIT_FAILURE) ;
	    }
	if(buf[0]!='%') break; // should finish line... and trim the left...
	if(1==sscanf(buf,"%% base %i\n",&base)) header_specify_base=true;
    }
    if(3!=sscanf(buf,"%u %u %u",&nrows,&ncols,&nnz))
	{
	    printf("Format problem in MatrixMarket file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    if(nrows!=ncols){
	printf("Dimension problem in MatrixMarket file %s\n",file);
	exit(EXIT_FAILURE) ;
    }

    *n=nrows;
    *total=nnz;
    *rows_index=(int *)malloc(((*n)+1)*sizeof(int));
    *cols=(int *)malloc((*total)*sizeof(int));
    *coefs=(T *)malloc((*total)*sizeof(T));

    std::vector<std::map<int,double> > row_map(nrows);

    int row,col;
    double val;

    if(2!=fscanf(fdata,"%u %u",&row,&col))
	{
	    if(feof(fdata)!=0)
		{
		    printf("EOF encoutered in %s while reading entry\n", file);
		    return;
		}
	    else
		{
		    printf("Format problem in MatrixMarket file %s\n",file);
		    exit(EXIT_FAILURE) ;
		}
	}

    if(!header_specify_base && row==1)
	{
	    base=1;
	}
    else
	{
	    if(!header_specify_base && row!=0)
		{
		    printf("Base problem in MatrixMarket file %s\n",file);
		    exit(EXIT_FAILURE) ;
		}
	}
    if(1!=fscanf(fdata,"%lf",&val))
	{
	    printf("Format problem in MatrixMarket file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}
    row_map[row-base].insert(std::pair<int,double>(col-base,val));
    if(is_symmetric && row!=col)
	row_map[col-base].insert(std::pair<int,double>(row-base,val));

    for(int i=1;i<nnz;++i) {
	if(2!=fscanf(fdata,"%u %u",&row,&col)) {
	    if(feof(fdata)!=0)
		{
		    printf("EOF encoutered in %s while reading entry\n", file);
		    return;
		}
	    else
		{
		    printf("Format problem in MatrixMarket file %s\n",file);
		    exit(EXIT_FAILURE) ;
		}
	}
	if(1!=fscanf(fdata,"%lf",&val))
	    {
		printf("Format problem in MatrixMarket file %s\n",file);
		exit(EXIT_FAILURE) ;
	    }
	row_map[row-base].insert(std::pair<int,double>(col-base,val));
	if(is_symmetric && row!=col)
	    row_map[col-base].insert(std::pair<int,double>(row-base,val));
    }

    int count=0;
    (*rows_index)[0]=0;
    for(int i=0;i<nrows;++i)
	{
	    std::map<int,double>::iterator it,it_end;
	    for(it=row_map[i].begin(), it_end=row_map[i].end(); it!=it_end; ++it)
		{
		    (*cols)[count]=it->first;
		    (*coefs)[count]=(T)it->second;
		    count++;
		}
	    (*rows_index)[i+1]=count;
	}
    fclose(fdata);
}
#define TEMPLATE_INSTANTIATION(T) \
void LoadMatrixMarket_CSR(char* file, int* n, int* total, int** rows_index, int** cols, T** coefs)
MM_INSTANCIATE_TEMPLATE(TEMPLATE_INSTANTIATION)
#undef TEMPLATE_INSTANTIATION
////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void LoadVector(char* file, int* n, T** vec, int ascii=1)
{
    FILE *fdata;
    int nb;
    double val;

    if ((fdata=fopen(file,"r")) == NULL)
	{
	    printf("Format problem in vectort file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    if(1!=fscanf(fdata,"%u",&nb))
	{
	    if(feof(fdata)!=0)
		{
		    printf("EOF encoutered in %s while reading entry\n", file);
		    return;
		}
	    else
		{
		    printf("Format problem in vector file %s\n",file);
		    exit(EXIT_FAILURE) ;
		}
	}

    *n=nb;
    *vec=(T*)malloc((*n)*sizeof(T));

    if(ascii){
	for(int i=0;i<(*n);++i)
	    {
		if(1!=fscanf(fdata,"%lf",&val))
		    {
			printf("Format problem in vector file %s\n",file);
			exit(EXIT_FAILURE) ;
		    }
		(*vec)[i]=(T)val;
	    }
    }
    else
	{
	    fgetc(fdata);
	    double *temp=(double*)malloc((*n)*sizeof(double));
	    if((*n)!=fread(temp,sizeof(double),(*n),fdata))
		{
		    printf("Format problem in vector file %s\n",file);
		    exit(EXIT_FAILURE);
		}
	    for(int i=0;i<(*n);++i)
		{
		    (*vec)[i]=(T)temp[i];
		}
	    free(temp);
	}
    fclose(fdata);
}
#define TEMPLATE_INSTANTIATION(T) \
void LoadVector(char* file, int* n, T** vec, int ascii=1)
MM_INSTANCIATE_TEMPLATE(TEMPLATE_INSTANTIATION)
#undef TEMPLATE_INSTANTIATION
////////////////////////////////////////////////////////////////////////////////////////////
//complex
////////////////////////////////////////////////////////////////////////////////////////////
template <class T2,class T>
void LoadMatrixMarket(char* file, int* n, int* total, int** rows, int** cols, _cudacomplex<T2,T>** coefs)
{
    FILE *fdata;
    int base = 0;
    bool header_specify_base=false;
    int nrows, ncols, nnz;
    const int BUF_SIZE = 256;
    char buf[BUF_SIZE];
    char float_type[32];
    char pattern_type[32];

    if ((fdata=fopen(file,"r")) == NULL)
	{
	    printf("Impossible d'ouvrir le fichier %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    fscanf(fdata,"%%%%MatrixMarket matrix coordinate %31s %31s\n",
	   float_type,pattern_type);
    if(0!=strcmp(float_type,"complex"))
	{
	    printf("Real number type in MatrixMarket file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    bool is_symmetric = (0==strcmp(pattern_type,"symmetric"));
    if(!is_symmetric && 0!=strcmp(pattern_type,"general"))
	{
	    printf("Format problem in MatrixMarket file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    while(1) {
	if(NULL==fgets(buf,BUF_SIZE,fdata))
	    {
		printf("Format problem in MatrixMarket file %s\n",file);
		exit(EXIT_FAILURE) ;
	    }
	if(buf[0]!='%') break; // should finish line... and trim the left...
	if(1==sscanf(buf,"%% base %i\n",&base)) header_specify_base=true;
    }
    if(3!=sscanf(buf,"%u %u %u",&nrows,&ncols,&nnz))
	{
	    printf("Format problem in MatrixMarket file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    if(nrows!=ncols){
	printf("Dimension problem in MatrixMarket file %s\n",file);
	exit(EXIT_FAILURE) ;
    }

    *n=nrows;
    *total=nnz;
    *rows=(int *)malloc((*total)*sizeof(int));
    *cols=(int *)malloc((*total)*sizeof(int));
    *coefs=(_cudacomplex<T2,T> *)malloc((*total)*sizeof(_cudacomplex<T2,T>));

    int row,col;
    _cudacomplex<double2,double> val;

    if(2!=fscanf(fdata,"%u %u",&row,&col))
	{
	    if(feof(fdata)!=0)
		{
		    printf("EOF encoutered in %s while reading entry\n", file);
		    return;
		}
	    else
		{
		    printf("Format problem in MatrixMarket file %s\n",file);
		    exit(EXIT_FAILURE) ;
		}
	}

    if(!header_specify_base && row==1)
	{
	    base=1;
	}
    else
	{
	    if(!header_specify_base && row!=0)
		{
		    printf("Base problem in MatrixMarket file %s\n",file);
		    exit(EXIT_FAILURE) ;
		}
	}
    if(2!=fscanf(fdata,"%lf %lf",&val.real(),&val.imag()))
	{
	    printf("Format problem in MatrixMarket file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    int compt=0;
    (*rows)[compt]=row-base;
    (*cols)[compt]=col-base;
    (*coefs)[compt]=_cudacomplex<T2,T>( val.real(), val.imag() );
    compt++;
    if(is_symmetric && row!=col){
	(*rows)[compt]=col-base;
	(*cols)[compt]=row-base;
	(*coefs)[compt]=_cudacomplex<T2,T>( val.real(), val.imag() );
	compt++;
    }

    for(int i=1;i<nnz;++i) {
	if(2!=fscanf(fdata,"%u %u",&row,&col)) {
	    if(feof(fdata)!=0)
		{
		    printf("EOF encoutered in %s while reading entry\n", file);
		    return;
		}
	    else
		{
		    printf("Format problem in MatrixMarket file %s\n",file);
		    exit(EXIT_FAILURE) ;
		}
	}
	if(2!=fscanf(fdata,"%lf %lf",&val.real(),&val.imag()))
	    {
		printf("Format problem in MatrixMarket file %s\n",file);
		exit(EXIT_FAILURE) ;
	    }
	(*rows)[compt]=row-base;
	(*cols)[compt]=col-base;
	(*coefs)[compt]=_cudacomplex<T2,T>( val.real(), val.imag() );
	compt++;
	if(is_symmetric && row!=col){
	    (*rows)[compt]=col-base;
	    (*cols)[compt]=row-base;
	    (*coefs)[compt]=_cudacomplex<T2,T>( val.real(), val.imag() );
	    compt++;
	}
    }
    fclose(fdata);
}
#define TEMPLATE_INSTANTIATION(T2,T)					\
    void LoadMatrixMarket(char* file, int* n, int* total, int** rows, int** cols, _cudacomplex<T2,T>** coefs)
MM_INSTANCIATE_TEMPLATE_2(TEMPLATE_INSTANTIATION)
#undef TEMPLATE_INSTANTIATION
////////////////////////////////////////////////////////////////////////////////////////////
template <class T2,class T>
void LoadMatrixMarket_CSR(char* file, int* n, int* total, int** rows_index, int** cols, _cudacomplex<T2,T>** coefs)
{

    FILE *fdata;
    int base = 0;
    bool header_specify_base=false;
    int nrows, ncols, nnz;
    const int BUF_SIZE = 256;
    char buf[BUF_SIZE];
    char float_type[32];
    char pattern_type[32];

    if ((fdata=fopen(file,"r")) == NULL)
	{
	    printf("Impossible d'ouvrir le fichier %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    fscanf(fdata,"%%%%MatrixMarket matrix coordinate %31s %31s\n",
	   float_type,pattern_type);
    if(0!=strcmp(float_type,"complex"))
	{
	    printf("Real number type in MatrixMarket file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    bool is_symmetric = (0==strcmp(pattern_type,"symmetric"));
    if(!is_symmetric && 0!=strcmp(pattern_type,"general"))
	{
	    printf("Format problem in MatrixMarket file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    while(1) {
	if(NULL==fgets(buf,BUF_SIZE,fdata))
	    {
		printf("Format problem in MatrixMarket file %s\n",file);
		exit(EXIT_FAILURE) ;
	    }
	if(buf[0]!='%') break; // should finish line... and trim the left...
	if(1==sscanf(buf,"%% base %i\n",&base)) header_specify_base=true;
    }
    if(3!=sscanf(buf,"%u %u %u",&nrows,&ncols,&nnz))
	{
	    printf("Format problem in MatrixMarket file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    if(nrows!=ncols){
	printf("Dimension problem in MatrixMarket file %s\n",file);
	exit(EXIT_FAILURE) ;
    }

    *n=nrows;
    *total=nnz;
    *rows_index=(int *)malloc(((*n)+1)*sizeof(int));
    *cols=(int *)malloc((*total)*sizeof(int));
    *coefs=(_cudacomplex<T2,T> *)malloc((*total)*sizeof(_cudacomplex<T2,T>));

    std::vector<std::map<int,_cudacomplex<double2,double> > > row_map(nrows);

    int row,col;
    _cudacomplex<double2,double> val;

    if(2!=fscanf(fdata,"%u %u",&row,&col))
	{
	    if(feof(fdata)!=0)
		{
		    printf("EOF encoutered in %s while reading entry\n", file);
		    return;
		}
	    else
		{
		    printf("Format problem in MatrixMarket file %s\n",file);
		    exit(EXIT_FAILURE) ;
		}
	}

    if(!header_specify_base && row==1)
	{
	    base=1;
	}
    else
	{
	    if(!header_specify_base && row!=0)
		{
		    printf("Base problem in MatrixMarket file %s\n",file);
		    exit(EXIT_FAILURE) ;
		}
	}
    if(2!=fscanf(fdata,"%lf %lf",&val.real(),&val.imag()))
	{
	    printf("Format problem in MatrixMarket file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}
    row_map[row-base].insert(std::pair<int,_cudacomplex<double2,double> >(col-base,val));
    if(is_symmetric && row!=col)
	row_map[col-base].insert(std::pair<int,_cudacomplex<double2,double> >(row-base,val));

    for(int i=1;i<nnz;++i) {
	if(2!=fscanf(fdata,"%u %u",&row,&col)) {
	    if(feof(fdata)!=0)
		{
		    printf("EOF encoutered in %s while reading entry\n", file);
		    return;
		}
	    else
		{
		    printf("Format problem in MatrixMarket file %s\n",file);
		    exit(EXIT_FAILURE) ;
		}
	}
	if(2!=fscanf(fdata,"%lf %lf",&val.real(),&val.imag()))
	    {
		printf("Format problem in MatrixMarket file %s\n",file);
		exit(EXIT_FAILURE) ;
	    }
	row_map[row-base].insert(std::pair<int,_cudacomplex<double2,double> >(col-base,val));
	if(is_symmetric && row!=col)
	    row_map[col-base].insert(std::pair<int,_cudacomplex<double2,double> >(row-base,val));
    }

    int count=0;
    (*rows_index)[0]=0;
    for(int i=0;i<nrows;++i)
	{
	    std::map<int,_cudacomplex<double2,double> >::iterator it,it_end;
	    for(it=row_map[i].begin(), it_end=row_map[i].end(); it!=it_end; ++it)
		{
		    (*cols)[count]=it->first;
		    (*coefs)[count]=_cudacomplex<T2,T>( it->second.real(), it->second.imag() );
		    count++;
		}
	    (*rows_index)[i+1]=count;
	}
    fclose(fdata);
}
#define TEMPLATE_INSTANTIATION(T2,T)					\
    void LoadMatrixMarket_CSR(char* file, int* n, int* total, int** rows_index, int** cols, _cudacomplex<T2,T>** coefs)
MM_INSTANCIATE_TEMPLATE_2(TEMPLATE_INSTANTIATION)
#undef TEMPLATE_INSTANTIATION
////////////////////////////////////////////////////////////////////////////////////////////
template <class T2,class T>
void LoadVector(char* file, int* n, _cudacomplex<T2,T>** vec, int ascii=1)
{
    FILE *fdata;
    int nb;
    _cudacomplex<double2,double> val;

    if ((fdata=fopen(file,"r")) == NULL)
	{
	    printf("Format problem in vectort file %s\n",file);
	    exit(EXIT_FAILURE) ;
	}

    if(1!=fscanf(fdata,"%u",&nb))
	{
	    if(feof(fdata)!=0)
		{
		    printf("EOF encoutered in %s while reading entry\n", file);
		    return;
		}
	    else
		{
		    printf("Format problem in vector file %s\n",file);
		    exit(EXIT_FAILURE) ;
		}
	}

    *n=nb;
    *vec=(_cudacomplex<T2,T>*)malloc((*n)*sizeof(_cudacomplex<T2,T>));

    if(ascii){
	for(int i=0;i<(*n);++i)
	    {
		if(2!=fscanf(fdata,"%lf %lf",&val.real(),&val.imag()))
		    {
			printf("Format problem in vector file %s\n",file);
			exit(EXIT_FAILURE) ;
		    }
		(*vec)[i]=_cudacomplex<T2,T>( val.real(), val.imag() );
	    }
    }
    else
	{
	    fgetc(fdata);
	    _cudacomplex<double2,double> *temp=(_cudacomplex<double2,double>*)malloc((*n)*sizeof(_cudacomplex<double2,double>));
	    if((*n)!=fread(temp,sizeof(_cudacomplex<double2,double>),(*n),fdata))
		{
		    printf("Format problem in vector file %s\n",file);
		    exit(EXIT_FAILURE);
		}
	    for(int i=0;i<(*n);++i)
		{
		    (*vec)[i]=_cudacomplex<T2,T>( temp[i].real(), temp[i].imag() );
		}
	    free(temp);
	}
    fclose(fdata);
}
#define TEMPLATE_INSTANTIATION(T2,T)					\
    void LoadVector(char* file, int* n, _cudacomplex<T2,T>** vec, int ascii=1)
MM_INSTANCIATE_TEMPLATE_2(TEMPLATE_INSTANTIATION)
#undef TEMPLATE_INSTANTIATION
////////////////////////////////////////////////////////////////////////////////////////////
