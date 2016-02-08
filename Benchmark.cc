/// @copyright (c) 2007 CSIRO
/// Australia Telescope National Facility (ATNF)
/// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
/// PO Box 76, Epping NSW 1710, Australia
///
/// This file is part of the ASKAP software distribution.
///
/// The ASKAP software distribution is free software: you can redistribute it
/// and/or modify it under the terms of the GNU General Public License as
/// published by the Free Software Foundation; either version 2 of the License,
/// or (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program; if not, write to the Free Software
/// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
///
/// This program was modified so as to use it in the contest.
/// The last modification was on January 12, 2015.
///

// Include own header file first
#include "Benchmark.h"
#include "Stopwatch.h"

#pragma offload_attribute(push, target(mic))

// System includes
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <mkl.h>
#include <climits>
#include <omp.h>

#pragma offload_attribute(pop)


Benchmark::Benchmark()
    : next(1)
{
}

// Return a pseudo-random integer in the range 0..2147483647
// Based on an algorithm in Kernighan & Ritchie, "The C Programming Language"
int Benchmark::randomInt()
{
    const unsigned int maxint = std::numeric_limits<int>::max();
    next = next * 1103515245 + 12345;
    return ((unsigned int)(next / 65536) % maxint);
}

void Benchmark::init()
{
    // Initialize the data to be gridded
    u.resize(nSamples);
    v.resize(nSamples);
    w.resize(nSamples);
    samples.resize(nSamples * nChan);
    outdata.resize(nSamples * nChan);


    Coord rd;
    FILE * fp;
    if((fp = fopen("randnum.dat", "rb")) == NULL)
    {
        printf("cannot open file\n");
        return;
    }

    for(int i = 0; i < nSamples; i++)
    {
        if(fread(&rd, sizeof(Coord), 1, fp) != 1) {printf("Rand number read error!\n");}
        u[i] = baseline * rd - baseline / 2;
        if(fread(&rd, sizeof(Coord), 1, fp) != 1) {printf("Rand number read error!\n");}
        v[i] = baseline * rd - baseline / 2;
        if(fread(&rd, sizeof(Coord), 1, fp) != 1) {printf("Rand number read error!\n");}
        w[i] = baseline * rd - baseline / 2;

        for(int chan = 0; chan < nChan; chan++)
        {
            if(fread(&rd, sizeof(Coord), 1, fp) != 1) {printf("Rand number read error!\n");}
            samples[i * nChan + chan].data = rd;
            outdata[i * nChan + chan] = 0.0;
        }
    }
    fclose(fp);

    grid.resize(gSize * gSize);
    grid.assign(grid.size(), Value(0.0));

    // Measure frequency in inverse wavelengths
    std::vector<Coord> freq(nChan);

    for(int i = 0; i < nChan; i++)
    {
        freq[i] = (1.4e9 - 2.0e5 * Coord(i) / Coord(nChan)) / 2.998e8;
    }

    // Initialize convolution function and offsets
    initC(freq, cellSize, wSize, m_support, overSample, wCellSize, C);
    initCOffset(u, v, w, freq, cellSize, wCellSize, wSize, gSize,
                m_support, overSample);
}

void Benchmark::runGrid()
{
    gridKernel(m_support, C, grid, gSize);
}

/////////////////////////////////////////////////////////////////////////////////
// The next function is the kernel of the gridding.
// The data are presented as a vector. Offsets for the convolution function
// and for the grid location are precalculated so that the kernel does
// not need to know anything about world coordinates or the shape of
// the convolution function. The ordering of cOffset and iu, iv is
// random.
//
// Perform gridding
//
// data - values to be gridded in a 1D vector
// support - Total width of convolution function=2*support+1
// C - convolution function shape: (2*support+1, 2*support+1, *)
// cOffset - offset into convolution function per data point
// iu, iv - integer locations of grid points
// grid - Output grid: shape (gSize, *)
// gSize - size of one axis of grid
void Benchmark::gridKernel(const int support,
                           const std::vector<Value>& C,
                           std::vector<Value>& grid, const int gSize)
{
    const int sSize = 2 * support + 1;
    const int num = int(samples.size());//num is the size of samples;
    const int gSz = int(grid.size());// the size of grid
    Value *mic_grid = new Value[gSz];//temparay space for mic's calculate result

    //openMP function
    const int max_threads = omp_get_max_threads();
    omp_set_nested(true);

    //the cpu_scale_p of the whole calculation will give to cpu, others give to mic
    // CPU和MIC的比重， CPU要多得多，为何？
    const double cpu_scale_p = 0.86;
    // 计算CPU和MIC异构计算的任务量
    const int cpu_scale = int(cpu_scale_p * num); //It's that how many samples will give to cpu to calculate
    const int mic_scale = num - cpu_scale;
    printf("cpu calculate:%.3lf%%\nmic calculate:%.3lf%%\n", cpu_scale_p*100, (1.0 - cpu_scale_p)*100);

    //compute gind dind & cind and sort into struct array: index
    // 任务划分，两个数组
    Index *index = new Index[num];
    Index *cpu_index = &index[0];//from index[0] to index[cpu_scale - 1] belong to cpu calculate
    Index *mic_index = &index[cpu_scale];//from index[cpu_scale] to index[num - 1] belong to mic calculate
    // 为何要静态，也许是方便调参数
    // 并行地赋初值
    #pragma omp parallel for schedule(static) num_threads(max_threads)
    for(int i = 0; i < max_threads; i++)
    {
        int begin = i * num / max_threads, end = (i + 1) * num / max_threads;
        for(int dind = begin; dind < end; ++dind)
        {
            index[dind] = (Index) {samples[dind].iu + gSize *samples[dind].iv - support, samples[dind].cOffset, dind};
        }
    }

    //calculate the mininum index and the maxinum index of C
    //so that minus the size of C which need to transfer to mic
    //also calculate the mininum index and the maxinum index of mic_grid
    //so that minus the size of mic_grid which need to transfer back to cpu
    int C_begin_index = C.size(), C_end_index = 0, C_transfer_size = 0;
    int mic_grid_begin_index = gSz, mic_grid_end_index = 0, mic_grid_transfer_size = 0;

    //sort index array according to gind
    // 为了求Lower bound，划分数据集为不相关的区间。先排序
    // cpu部分的排序和mic部分的排序是并发的，因此放在两个section里
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            parallel_sort(0, mic_scale, mic_index, 3);
            if(mic_scale != 0)
            {
                mic_grid_begin_index = mic_index[0].gind - sSize * gSize;
                mic_grid_end_index = mic_index[mic_scale - 1].gind + sSize * gSize;
                mic_grid_transfer_size = mic_grid_end_index - mic_grid_begin_index;
            }
        }
        #pragma omp section
        {parallel_sort(0, cpu_scale, cpu_index, 3);}
    }

    //divide index into groups according to gind's row index
    int *mic_index_group = new int[gSize + 1];
    int *cpu_index_group = new int[gSize + 1];
    // 计算lower_bound。仍然是static
    // 这样的lower_bound看起来不合理，应该是保证两段之间的距离不小于gSize即可
    // 因此这个lower_bound可能应该是前后数据相关的
    #pragma omp parallel for schedule(static) num_threads(max_threads)
    for(int i = 0; i < max_threads; i++)
    {
        int begin = i * (gSize + 1) / max_threads,
            end = (i + 1) * (gSize + 1) / max_threads;
        for(int now_row = begin; now_row < end; now_row++)
        {
            mic_index_group[now_row] = std::lower_bound(mic_index, mic_index + mic_scale,
            (Index) {gSize*now_row, 0, 0}, cmp_gind) - mic_index;
            cpu_index_group[now_row] = std::lower_bound(cpu_index, cpu_index + cpu_scale,
            (Index) {gSize*now_row, 0, 0}, cmp_gind) - cpu_index;
        }
    }

    // 又划分任务
    int mic_begin_group_index = std::upper_bound(mic_index_group, mic_index_group + gSize + 1,
                                0) - mic_index_group;
    int mic_end_group_index = std::lower_bound(mic_index_group, mic_index_group + gSize + 1,
                              mic_scale) - mic_index_group + sSize;
    int mic_group_size = mic_end_group_index - mic_begin_group_index - sSize;
    int cpu_begin_group_index = std::upper_bound(cpu_index_group, cpu_index_group + gSize + 1,
                                0) - cpu_index_group;
    int cpu_end_group_index = std::lower_bound(cpu_index_group, cpu_index_group + gSize + 1,
                              cpu_scale) - cpu_index_group + sSize;
    int cpu_group_size = cpu_end_group_index - cpu_begin_group_index - sSize;


    //sort a row according to cind
    // 为什么还要在组内按照cind排序呢？
    #pragma omp parallel for schedule(static) num_threads(max_threads)
    for(int i = 0; i < max_threads; i++)
    {
        int begin = i * mic_group_size / max_threads + mic_begin_group_index;
        int end = (i + 1) * mic_group_size / max_threads + mic_begin_group_index;
        for(int now_row = begin; now_row < end; now_row++)
        {
            std::sort(&mic_index[mic_index_group[now_row]], &mic_index[mic_index_group[now_row + 1]], cmp_cind);
        }
        begin = i * cpu_group_size / max_threads + cpu_begin_group_index;
        end = (i + 1) * cpu_group_size / max_threads + cpu_begin_group_index;
        for(int now_row = begin; now_row < end; now_row++)
        {
            std::sort(&cpu_index[cpu_index_group[now_row]], &cpu_index[cpu_index_group[now_row + 1]], cmp_cind);
        }
    }


    // 为什么要拷贝呢？
    // copy data from samples.data
    Value *data = new Value[num];
    #pragma omp parallel for schedule(static) num_threads(max_threads)
    for(int i = 0; i < max_threads; i++)
    {
        int begin = i * num / max_threads, end = (i + 1) * num / max_threads;
        cblas_zcopy(end - begin, &samples[begin], 2, &data[begin], 1);
    }


    //calaulte C_begin_index and C_end_index
    for(int i = mic_begin_group_index; i < mic_end_group_index - sSize; i++)
    {
        if(mic_index[mic_index_group[i]].cind < C_begin_index)
            C_begin_index = mic_index[mic_index_group[i]].cind;
        if(mic_index[mic_index_group[i + 1] - 1].cind > C_end_index)
            C_end_index = mic_index[mic_index_group[i + 1] - 1].cind;
    }
    C_transfer_size = C_end_index - C_begin_index + sSize * sSize;

    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        if(mic_scale != 0)//if mic_scale == 0 dont'need calculate in mic
        {
            /********** mic calculate part ************/
            //pointer for offload
            double *pC = (double *)&C[0];
            double *pgrid = (double *)&mic_grid[0];
            double *pdata = (double *)&data[0];

            //mic data transfer
#pragma offload_transfer target(mic)\
				in(mic_index:length(mic_scale) free_if(0))\
				in(mic_index_group:length(gSize + 1) free_if(0))\
				in(pC[C_begin_index*2:C_transfer_size*2]: free_if(0))\
				in(pdata:length(num*2) free_if(0))\
				nocopy(pgrid)

#pragma offload target(mic)\
				nocopy(mic_index:free_if(1))\
				nocopy(mic_index_group:free_if(1))\
				nocopy(pC:free_if(1))\
				nocopy(pdata:free_if(1))\
				out(pgrid[mic_grid_begin_index*2:mic_grid_transfer_size*2]: free_if(1))
            {
                Value *mC = (Value*)pC;
                Value *data = (Value*)pdata;
                Value *mic_grid = (Value*)pgrid;

                // 对各个分组的所有样本进行计算
                #pragma omp parallel for schedule(dynamic)
                for(int now_row = mic_begin_group_index; now_row < mic_end_group_index; now_row++)
                {
                    int gind_add = gSize * sSize;
                    int cind_add = sSize * sSize;

                    for(int now_group = now_row - sSize + 1; now_group <= now_row; now_group++)
                    {
                        gind_add -= gSize;
                        cind_add -= sSize;

                        for(int now_index = mic_index_group[now_group - 1]; now_index < mic_index_group[now_group]; now_index++)
                        {
                            // y = a*x + y的库函数
                            cblas_zaxpy(sSize, &data[mic_index[now_index].dind], &mC[mic_index[now_index].cind + cind_add], 1, &mic_grid[mic_index[now_index].gind + gind_add], 1);
                        }
                    }
                }
            }
        }
        // CPU部分的运算，这部分和前面部分为何要用omp section？？
        #pragma omp section
        if(cpu_scale != 0)//if cpu_scale == 0, that's cpu don't need calculate
        {
            /*********** cpu calculate part ****************/
            // 和MIC端代码几乎一样
            #pragma omp parallel for schedule(dynamic) num_threads(max_threads)
            for(int now_row = cpu_begin_group_index; now_row < cpu_end_group_index; now_row++)
            {
                int gind_add = gSize * sSize;
                int cind_add = sSize * sSize;

                for(int now_group = now_row - sSize + 1; now_group <= now_row; now_group++)
                {
                    gind_add -= gSize;
                    cind_add -= sSize;

                    for(int now_index = cpu_index_group[now_group - 1]; now_index < cpu_index_group[now_group]; now_index++)
                    {
                        cblas_zaxpy(sSize, &data[cpu_index[now_index].dind], &C[cpu_index[now_index].cind + cind_add], 1, &grid[cpu_index[now_index].gind + gind_add], 1);
                    }
                }
            }
        }
    }
    //gather the result
    // gather结果
    Value a(1.0, 0.0);
    #pragma omp parallel for schedule(static) num_threads(max_threads)
    for(int i = 0; i < max_threads; i++)
    {
        int begin = i * mic_grid_transfer_size / max_threads + mic_grid_begin_index,
            end = (i + 1) * mic_grid_transfer_size / max_threads + mic_grid_begin_index;
        cblas_zaxpy(end - begin, &a, &mic_grid[begin], 1, &grid[begin], 1);
    }
}
inline bool cmp_gind(Index a, Index b)
{
    return a.gind < b.gind;
}
inline bool cmp_cind(Index a, Index b)
{
    return a.cind < b.cind;
}

// 并行的归并排序
void parallel_sort(int l, int r, Index *index, int parallel_num)
{
    if(!parallel_num)
    {
        std::sort(&index[l], &index[r], cmp_gind);
        return ;
    }
    else
    {
        int Sz = r - l;
        if(Sz <= 0)
            return ;
        int Sz1 = (Sz >> 1), Sz2 = Sz - Sz1;
        // 使用递归效率应该会低
        // 还是双调排序更好
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                parallel_sort(l, l + Sz1, index, parallel_num - 1);
            }
            #pragma omp section
            {
                parallel_sort(l + Sz1, r, index, parallel_num - 1);
            }
        }
        Index *tmp1 = new Index[Sz1];
        Index *tmp2 = new Index[Sz2];
        for(int i = 0; i < Sz1; i++)
        {
            tmp1[i] = index[i + l];
        }
        for(int i = 0; i < Sz2; i++)
        {
            tmp2[i] = index[i + l + Sz1];
        }
        //merge
        int i = l, i1 = 0, i2 = 0;
        while(1)
        {
            if(tmp1[i1].gind < tmp2[i2].gind)
            {
                index[i++] = tmp1[i1++];
                if(i1 == Sz1)
                {
                    while(i < r)index[i++] = tmp2[i2++];
                    return ;
                }
            }
            else
            {
                index[i++] = tmp2[i2++];
                if(i2 == Sz2)
                {
                    while(i < r) index[i++] = tmp1[i1++];
                    return ;
                }
            }
        }
    }
}
/////////////////////////////////////////////////////////////////////////////////
// Initialize W project convolution function
// - This is application specific and should not need any changes.
//
// freq - temporal frequency (inverse wavelengths)
// cellSize - size of one grid cell in wavelengths
// wSize - Size of lookup table in w
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
void Benchmark::initC(const std::vector<Coord>& freq,
                      const Coord cellSize, const int wSize,
                      int& support, int& overSample,
                      Coord& wCellSize, std::vector<Value>& C)
{
    std::cout << "Initializing W projection convolution function" << std::endl;
    support = static_cast<int>(1.5 * sqrt(std::abs(baseline) * static_cast<Coord>(cellSize)
                                          * freq[0]) / cellSize);

    overSample = 8;
    std::cout << "Support = " << support << " pixels" << std::endl;
    wCellSize = 2 * baseline * freq[0] / wSize;
    std::cout << "W cellsize = " << wCellSize << " wavelengths" << std::endl;

    // Convolution function. This should be the convolution of the
    // w projection kernel (the Fresnel term) with the convolution
    // function used in the standard case. The latter is needed to
    // suppress aliasing. In practice, we calculate entire function
    // by Fourier transformation. Here we take an approximation that
    // is good enough.
    const int sSize = 2 * support + 1;

    const int cCenter = (sSize - 1) / 2;

    C.resize(sSize * sSize * overSample * overSample * wSize);
    std::cout << "Size of convolution function = " << sSize*sSize*overSample
              *overSample*wSize*sizeof(Value) / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Shape of convolution function = [" << sSize << ", " << sSize << ", "
              << overSample << ", " << overSample << ", " << wSize << "]" << std::endl;

    for(int k = 0; k < wSize; k++)
    {
        double w = double(k - wSize / 2);
        double fScale = sqrt(std::abs(w) * wCellSize * freq[0]) / cellSize;

        for(int osj = 0; osj < overSample; osj++)
        {
            for(int osi = 0; osi < overSample; osi++)
            {
                for(int j = 0; j < sSize; j++)
                {
                    double j2 = std::pow((double(j - cCenter) + double(osj) / double(overSample)), 2);

                    for(int i = 0; i < sSize; i++)
                    {
                        double r2 = j2 + std::pow((double(i - cCenter) + double(osi) / double(overSample)), 2);
                        long int cind = i + sSize * (j + sSize * (osi + overSample * (osj + overSample * k)));

                        if(w != 0.0)
                        {
                            C[cind] = static_cast<Value>(std::cos(r2 / (w * fScale)));
                        }
                        else
                        {
                            C[cind] = static_cast<Value>(std::exp(-r2));
                        }
                    }
                }
            }
        }
    }

    // Now normalise the convolution function
    Coord sumC = 0.0;

    for(int i = 0; i < sSize * sSize * overSample * overSample * wSize; i++)
    {
        sumC += std::abs(C[i]);
    }

    for(int i = 0; i < sSize * sSize * overSample * overSample * wSize; i++)
    {
        C[i] *= Value(wSize * overSample * overSample / sumC);
    }
}

// Initialize Lookup function
// - This is application specific and should not need any changes.
//
// freq - temporal frequency (inverse wavelengths)
// cellSize - size of one grid cell in wavelengths
// gSize - size of grid in pixels (per axis)
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
// wSize - Size of lookup table in w
void Benchmark::initCOffset(const std::vector<Coord>& u, const std::vector<Coord>& v,
                            const std::vector<Coord>& w, const std::vector<Coord>& freq,
                            const Coord cellSize, const Coord wCellSize,
                            const int wSize, const int gSize, const int support,
                            const int overSample)
{
    const int nSamples = u.size();
    const int nChan = freq.size();

    const int sSize = 2 * support + 1;

    // Now calculate the offset for each visibility point
    for(int i = 0; i < nSamples; i++)
    {
        for(int chan = 0; chan < nChan; chan++)
        {

            int dind = i * nChan + chan;

            Coord uScaled = freq[chan] * u[i] / cellSize;
            samples[dind].iu = int(uScaled);

            if(uScaled < Coord(samples[dind].iu))
            {
                samples[dind].iu -= 1;
            }

            int fracu = int(overSample * (uScaled - Coord(samples[dind].iu)));
            samples[dind].iu += gSize / 2;

            Coord vScaled = freq[chan] * v[i] / cellSize;
            samples[dind].iv = int(vScaled);

            if(vScaled < Coord(samples[dind].iv))
            {
                samples[dind].iv -= 1;
            }

            int fracv = int(overSample * (vScaled - Coord(samples[dind].iv)));
            samples[dind].iv += gSize / 2;

            // The beginning of the convolution function for this point
            Coord wScaled = freq[chan] * w[i] / wCellSize;
            int woff = wSize / 2 + int(wScaled);
            samples[dind].cOffset = sSize * sSize * (fracu + overSample * (fracv + overSample * woff));
        }
    }
}

void Benchmark::printGrid()
{
    FILE * fp;
    if((fp = fopen("grid.dat", "wb")) == NULL)
    {
        printf("cannot open file\n");
        return;
    }

    unsigned ij;
    for(int i = 0; i < gSize; i++)
    {
        for(int j = 0; j < gSize; j++)
        {
            ij = j + i * gSize;
            if(fwrite(&grid[ij], sizeof(Value), 1, fp) != 1)
                printf("File write error!\n");

        }
    }

    fclose(fp);
}

int Benchmark::getSupport()
{
    return m_support;
};
