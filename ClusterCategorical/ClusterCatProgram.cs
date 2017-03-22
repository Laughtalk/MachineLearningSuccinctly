using System;

namespace ClusterCategorical
{
    class ClusterCatProgram
    {
        /// <summary>
        /// 非数字化数据分组
        /// </summary>
        /// <param name="args">系统变量</param>
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin categorical data clustering demo\n");
            string[][] rawData = new string[7][];
            rawData[0] = new string[] { "Blue", "Small", "False" };
            rawData[1] = new string[] { "Green", "Medium", "True" };
            rawData[2] = new string[] { "Red", "Large", "False" };
            rawData[3] = new string[] { "Red", "Small", "True" };
            rawData[4] = new string[] { "Green", "Medium", "False" };
            rawData[5] = new string[] { "Yellow", "Medium", "False" };
            rawData[6] = new string[] { "Red", "Large", "False" };

            Console.WriteLine("Raw unclustered data:\n");
            Console.WriteLine(" Color Size Heavy");
            Console.WriteLine("-----------------------------");
            ShowData(rawData);

            int numClusters = 2;
            Console.WriteLine("\nSetting numClusters to " + numClusters);
            int numRestarts = 4;
            Console.WriteLine("Setting numRestarts to " + numRestarts);
            Console.WriteLine("\nStarting clustering using greedy category utility");
            CatClusterer cc = new CatClusterer(numClusters, rawData); // restart version
            double cu;
            int[] clustering = cc.Cluster(numRestarts, out cu);
            Console.WriteLine("Clustering complete\n");
            Console.WriteLine("Final clustering in internal form:");
            ShowVector(clustering, true);
            Console.WriteLine("Final CU value = " + cu.ToString("F4"));
            Console.WriteLine("\nRaw data grouped by cluster:\n");
            ShowClustering(numClusters, clustering, rawData);
            Console.WriteLine("\nEnd categorical data clustering demo\n");
            Console.ReadLine();
        }
        /// <summary>
        /// 显示二维字符串数组
        /// </summary>
        /// <param name="matrix">数据</param>
        static void ShowData(string[][] matrix) // for tuples
        {
            for (int i = 0; i < matrix.Length; ++i)
            {
                Console.Write("[" + i + "] ");
                for (int j = 0; j < matrix[i].Length; ++j)
                    Console.Write(matrix[i][j].ToString().PadRight(8) + " ");
                Console.WriteLine("");
            }
        }
        /// <summary>
        /// 显示一维整数数组
        /// </summary>
        /// <param name="vector">一维数组</param>
        /// <param name="newLine">是否换行</param>
        public static void ShowVector(int[] vector, bool newLine) // for clustering
        {
            for (int i = 0; i < vector.Length; ++i)
                Console.Write(vector[i] + " ");
            Console.WriteLine("");
            if (newLine == true)
                Console.WriteLine("");
        }
        /// <summary>
        /// 显示分组情况
        /// </summary>
        /// <param name="numClusters">总组数</param>
        /// <param name="clustering">分组情况数组</param>
        /// <param name="rawData">各个tuple</param>
        static void ShowClustering(int numClusters, int[] clustering, string[][] rawData)
        {
            Console.WriteLine("-----------------------------");
            for (int k = 0; k < numClusters; ++k) // display by cluster
            {
                for (int i = 0; i < rawData.Length; ++i) // each tuple
                {
                    if (clustering[i] == k) // curr tuple i belongs to curr cluster k
                    {
                        Console.Write(i.ToString().PadLeft(2) + " ");
                        for (int j = 0; j < rawData[i].Length; ++j)
                        {
                            Console.Write(rawData[i][j].ToString().PadRight(8) + " ");
                        }
                        Console.WriteLine("");
                    }
                }
                Console.WriteLine("-----------------------------");
            }
        }
    } // ClusterCatProgram
}
