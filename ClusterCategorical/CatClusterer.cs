using System;
using System.Collections.Generic;

namespace ClusterCategorical
{
    class CatClusterer
    {
        private int numClusters; // number of clusters
        private int[] clustering; // index = a tuple, value = cluster ID
        private int[][] dataAsInts; // ex: red = 0, blue = 1, green = 2
        private int[][][] valueCounts; // scratch to compute CU [att][val][count](sum)
        private int[] clusterCounts; // number tuples assigned to each cluster (sum)
        private int[] maxIndex; // max index of every attribute
        private Random rnd; // for several randomizations
        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="numClusters">分组组数</param>
        /// <param name="rawData">原数据</param>
        public CatClusterer(int numClusters, string[][] rawData)
        {
            this.numClusters = numClusters;
            MakeDataMatrix(rawData); // convert strings to ints into this.dataAsInts[][]
            Allocate(); // allocate all arrays & matrices (no initialize values)
        }
        /// <summary>
        /// 分类函数
        /// </summary>
        /// <param name="numRestarts">尝试次数。可设置为数据数量的平方根</param>
        /// <param name="catUtility">最终的CU值</param>
        /// <returns>返回最终分类结果</returns>
        public int[] Cluster(int numRestarts, out double catUtility)
        {
            // restart version
            int numRows = dataAsInts.Length;
            double currCU, bestCU = 0.0;
            int[] bestClustering = new int[numRows];
            for (int start = 0; start < numRestarts; ++start)
            {
                int seed = start; // use the start index as rnd seed
                int[] currClustering = ClusterOnce(seed, out currCU);
                if (currCU > bestCU)
                {
                    bestCU = currCU;
                    Array.Copy(currClustering, bestClustering, numRows);
                }
            }
            catUtility = bestCU;
            return bestClustering;
        }
        /// <summary>
        /// 一次分类
        /// </summary>
        /// <param name="seed">随机数种子</param>
        /// <param name="catUtility">单次尝试的CU值</param>
        /// <returns>单次尝试的分类结果</returns>
        private int[] ClusterOnce(int seed, out double catUtility)
        {
            this.rnd = new Random(seed);
            Initialize(); // clustering[] to -1, all counts[] to 0
            int[] goodIndexes = GetGoodIndices(dataAsInts.Length); // tuples that are dissimilar, param is times of trying

            for (int k = 0; k < numClusters; ++k) // assign first tuples to clusters
                Assign(goodIndexes[k], k);

            int numRows = dataAsInts.Length;
            int[] rndSequence = new int[numRows];
            for (int i = 0; i < numRows; ++i)
                rndSequence[i] = i;
            Shuffle(ref rndSequence); // present tuples in random sequence
            for (int t = 0; t < numRows; ++t) // main loop. walk through each tuple
            {
                int idx = rndSequence[t]; // index of data tuple to process
                if (clustering[idx] != -1) continue; // tuple clustered already
                double[] candidateCU = new double[numClusters]; // candidate CU values
                for (int k = 0; k < numClusters; ++k) // examine each cluster
                {
                    Assign(idx, k); // tentative cluster assignment
                    candidateCU[k] = CategoryUtility(); // compute and save the CU
                    Unassign(idx, k); // undo tentative assignment
                }
                int bestK = IndexOfMax(candidateCU); // greedy. the index is a cluster ID
                Assign(idx, bestK); // now we know which cluster gave the best CU
            }
            catUtility = CategoryUtility();
            int[] result = new int[numRows];
            Array.Copy(this.clustering, result, numRows);
            return result;
        }
        /// <summary>
        /// 用字典把字符串类型的数据转为整形
        /// </summary>
        /// <param name="rawData">原二维字符串数组</param>
        private void MakeDataMatrix(string[][] rawData)
        {
            //初始化dataAsInts[][]
            int numRows = rawData.Length;
            int numCols = rawData[0].Length;
            this.dataAsInts = new int[numRows][]; // allocate all
            this.maxIndex = new int[numCols];
            for (int i = 0; i < numRows; ++i)
                dataAsInts[i] = new int[numCols];
            //用字典把字符串类型的数据转为整形，方便构造后续数据结构
            for (int col = 0; col < numCols; ++col)
            {
                int idx = 0;
                Dictionary<string, int> dict = new Dictionary<string, int>();
                for (int row = 0; row < numRows; ++row) // build dict for curr col
                {
                    if (dict.ContainsKey(rawData[row][col]) == false)
                        dict.Add(rawData[row][col], idx++);
                    this.dataAsInts[row][col] = dict[rawData[row][col]];
                }
                maxIndex[col] = idx;
            }
            return; // explicit return style
        }
        /// <summary>
        /// 分配空间
        /// </summary>
        private void Allocate()
        {
            // assumes dataAsInts has been created
            // allocate this.clustering[], this.clusterCounts[], this.valueCounts[][][]
            int numRows = dataAsInts.Length;
            int numCols = dataAsInts[0].Length;
            this.clustering = new int[numRows];
            this.clusterCounts = new int[numClusters + 1]; // last cell is for sum

            this.valueCounts = new int[numCols][][]; // 1st dim
            for (int col = 0; col < numCols; ++col) // need # distinct values in each col
                this.valueCounts[col] = new int[maxIndex[col] + 1][]; // 0-based 2nd dim
            for (int i = 0; i < this.valueCounts.Length; ++i) // 3rd dim
                for (int j = 0; j < this.valueCounts[i].Length; ++j)
                    this.valueCounts[i][j] = new int[numClusters + 1]; // +1 last cell is for sum
            return;
        }
        /// <summary>
        /// 初始化 clustering[] to -1, all counts[] to 0
        /// </summary>
        private void Initialize()
        {
            for (int i = 0; i < clustering.Length; ++i)
                clustering[i] = -1;
            for (int i = 0; i < clusterCounts.Length; ++i)
                clusterCounts[i] = 0;
            for (int i = 0; i < valueCounts.Length; ++i)
                for (int j = 0; j < valueCounts[i].Length; ++j)
                    for (int k = 0; k < valueCounts[i][j].Length; ++k)
                        valueCounts[i][j][k] = 0;
            return;
        }
        /// <summary>
        /// 计算当前CU值。called by ClusterOnce
        /// </summary>
        /// <returns>返回当前CU值</returns>
        private double CategoryUtility()
        {
            // (E P(C[k]) * ( EE P(A[i]=V[i][j] | C[k])^2 - EE P(A[i]=V[i][j])^2 ] ) / n
            // because CU is called many times use precomputed counts 
            int numTuplesAssigned = clusterCounts[clusterCounts.Length - 1]; // last cell
            double[] clusterProbs = new double[this.numClusters];
            // calculate P(C[k])
            for (int k = 0; k < numClusters; ++k)
                clusterProbs[k] = (double)clusterCounts[k] / (double)numTuplesAssigned;
            // calculate single unconditional prob term (EE P(A[i]=V[i][j])^2)
            double unconditional = 0.0;
            for (int i = 0; i < valueCounts.Length; ++i)
            {
                for (int j = 0; j < valueCounts[i].Length; ++j)
                {
                    double p = (double)valueCounts[i][j][numClusters] / (double)numTuplesAssigned;// last cell holds sum
                    unconditional += (p * p);
                }
            }
            // calculate conditional terms each cluster (EE P(A[i]=V[i][j] | C[k])^2)
            double[] conditionals = new double[numClusters];
            for (int k = 0; k < numClusters; ++k)
            {
                for (int i = 0; i < valueCounts.Length; ++i) // each att
                {
                    for (int j = 0; j < valueCounts[i].Length; ++j) // each value
                    {
                        double p = (double)valueCounts[i][j][k] / (double)clusterCounts[k];
                        conditionals[k] += (p * p);
                    }
                }
            }
            // we have P(C[k]), EE P(A[i]=V[i][j] | C[k])^2, EE P(A[i]=V[i][j])^2 so we can compute CU easily
            double summation = 0.0;
            for (int k = 0; k < numClusters; ++k)
                summation += clusterProbs[k] * (conditionals[k] - unconditional);
            // (E P(C[k]) * ( EE P(A[i]=V[i][j] | C[k])^2 - EE P(A[i]=V[i][j])^2 ] ) / n
            return summation / numClusters;
        }
        /// <summary>
        /// 找出数组中最大值，返回最大值的下标
        /// </summary>
        /// <param name="cus">传入浮点数数组</param>
        /// <returns>返回整形下标</returns>
        private static int IndexOfMax(double[] cus)
        {
            // helper for ClusterOnce. returns index of largest value in array
            double bestCU = 0.0;
            int indexOfBestCU = 0;
            for (int k = 0; k < cus.Length; ++k)
            {
                if (cus[k] > bestCU)
                {
                    bestCU = cus[k];
                    indexOfBestCU = k;
                }
            }
            return indexOfBestCU;
        }
        /// <summary>
        /// Fisher-Yates洗牌算法
        /// 将indices中的项目乱序
        /// </summary>
        /// <param name="indices"></param>
        private void Shuffle(ref int[] indices) // instance so can use class rnd
        {
            for (int i = 0; i < indices.Length; ++i) // Fisher-Yates shuffle
            {
                int ri = rnd.Next(i, indices.Length); // random index
                int tmp = indices[i];
                indices[i] = indices[ri]; // swap
                indices[ri] = tmp;
            }
        }
        /// <summary>
        /// assign tuple at dataIndex to clustering[] cluster
        /// update valueCounts[][][], clusterCounts[]
        /// </summary>
        /// <param name="dataIndex">数据下标</param>
        /// <param name="clusterID">分组组号</param>
        private void Assign(int dataIndex, int clusterID)
        {
            clustering[dataIndex] = clusterID; // assign
            for (int i = 0; i < valueCounts.Length; ++i) // update valueCounts
            {
                int v = dataAsInts[dataIndex][i]; // att value
                ++valueCounts[i][v][clusterID]; // bump count
                ++valueCounts[i][v][numClusters]; // bump sum
            }
            ++clusterCounts[clusterID]; // update clusterCounts
            ++clusterCounts[numClusters]; // last cell is sum
        }
        /// <summary>
        /// unassign tuple at dataIndex to clustering[] cluster
        /// update valueCounts[][][], clusterCounts[]
        /// </summary>
        /// <param name="dataIndex"></param>
        /// <param name="clusterID"></param>
        private void Unassign(int dataIndex, int clusterID)
        {
            clustering[dataIndex] = -1; // unassign
            for (int i = 0; i < valueCounts.Length; ++i) // update
            {
                int v = dataAsInts[dataIndex][i];
                --valueCounts[i][v][clusterID];
                --valueCounts[i][v][numClusters]; // last cell is sum
            }
            --clusterCounts[clusterID]; // update clusterCounts
            --clusterCounts[numClusters]; // last cell
        }
        /// <summary>
        /// 获取相似性较低的初始项
        /// </summary>
        /// <param name="numTrials"></param>
        /// <returns></returns>
        private int[] GetGoodIndices(int numTrials)
        {
            // return numClusters indices of tuples that are different
            int numRows = dataAsInts.Length;
            int numCols = dataAsInts[0].Length;
            int[] result = new int[numClusters];
            int largestDiff = -1; // differences for a set of numClusters tuples
            for (int trial = 0; trial < numTrials; ++trial)
            {
                int[] candidates = Reservoir(numClusters, numRows);
                int numDifferences = 0; // for these candidates
                /*
                 * 两种选择较大差值初始项的方式
                 * 一种比较了candidates中所有的可能配对（当前使用）
                 * 一种只比较candidates中的相邻项（注释起来了）
                 */
                for (int i = 0; i < candidates.Length - 1; ++i) // all possible pairs
                {
                    for (int j = i + 1; j < candidates.Length; ++j)
                    {
                        int aRow = candidates[i];
                        int bRow = candidates[j];
                        for (int col = 0; col < numCols; ++col)
                            if (dataAsInts[aRow][col] != dataAsInts[bRow][col])
                                ++numDifferences;
                    }
                }
                //for (int i = 0; i < candidates.Length - 1; ++i) // only adjacent pairs
                //{
                //    int aRow = candidates[i];
                //    int bRow = candidates[i + 1];
                //    for (int col = 0; col < numCols; ++col)
                //        if (dataAsInts[aRow][col] != dataAsInts[bRow][col])
                //            ++numDifferences;
                //}
                if (numDifferences > largestDiff)
                {
                    largestDiff = numDifferences;
                    Array.Copy(candidates, result, numClusters);
                }
            }
            return result;
        }
        /// <summary>
        /// 返回n个位于[0, range)之间的随机数
        /// </summary>
        /// <param name="n">n个随机数</param>
        /// <param name="range">位于[0, range)之间</param>
        /// <returns>返回随机数列</returns>
        private int[] Reservoir(int n, int range) // helper for GetGoodIndices
        {
            // select n random indices between [0, range)
            int[] result = new int[n];
            for (int i = 0; i < n; ++i)
                result[i] = i;
            for (int t = n; t < range; ++t)
            {
                int j = rnd.Next(0, t + 1);
                if (j < n)
                    result[j] = t;
            }
            return result;
        }
    }
}
