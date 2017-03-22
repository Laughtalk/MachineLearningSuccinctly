using System;

namespace ClusterNumeric
{
    class ClusterProgram
    {
        /// <summary>
        /// K-Mean算法
        /// </summary>
        /// <param name="args">系统变量</param>
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin k-means clustering demo\n");
            double[][] rawData = new double[10][];
            rawData[0] = new double[] { 73, 72.6 };
            rawData[1] = new double[] { 61, 54.4 };
            rawData[2] = new double[] { 67, 99.9 };
            rawData[3] = new double[] { 68, 97.3 };
            rawData[4] = new double[] { 62, 59.0 };
            rawData[5] = new double[] { 75, 81.6 };
            rawData[6] = new double[] { 74, 77.1 };
            rawData[7] = new double[] { 66, 97.3 };
            rawData[8] = new double[] { 68, 93.3 };
            rawData[9] = new double[] { 61, 59.0 };

            //double[][] rawData = LoadData("..\\..\\HeightWeight.txt", 10, 2, ',');
            Console.WriteLine("Raw unclustered height (in.) weight (kg.) data:\n");
            Console.WriteLine(" ID Height Weight");
            Console.WriteLine("---------------------");
            ShowData(rawData, 1, true, true);

            int numClusters = 3;
            Console.WriteLine("\nSetting numClusters to " + numClusters);
            Console.WriteLine("Starting clustering using k-means algorithm");
            Clusterer c = new Clusterer(numClusters);
            int[] clustering = c.Cluster(rawData);
            Console.WriteLine("Clustering complete\n");

            Console.WriteLine("Final clustering in internal form:\n");
            ShowVector(clustering, true);
            Console.WriteLine("Raw data by cluster:\n");
            ShowClustered(rawData, clustering, numClusters, 1);
            Console.WriteLine("\nEnd k-means clustering demo\n");
            Console.ReadLine();
        }
        /// <summary>
        /// 显示二维数组中的浮点数
        /// </summary>
        /// <param name="data">用于显示的数据</param>
        /// <param name="decimals">保留的小数位数</param>
        /// <param name="indices">是否在最左面显示索引序号</param>
        /// <param name="newLine">是否需要末尾换行</param>
        static void ShowData(double[][] data, int decimals, bool indices, bool newLine)
        {
            for (int i = 0; i < data.Length; ++i)
            {
                if (indices == true)
                    Console.Write(i.ToString().PadLeft(3) + " ");
                for (int j = 0; j < data[i].Length; ++j)
                {
                    double v = data[i][j];
                    Console.Write(v.ToString("F" + decimals) + " ");
                }
                Console.WriteLine("");
            }
            if (newLine == true)
                Console.WriteLine("");
        }
        /// <summary>
        /// 显示一维数组中的整数
        /// </summary>
        /// <param name="vector">需要显示的整数数组</param>
        /// <param name="newLine">是否末尾换行</param>
        static void ShowVector(int[] vector, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
                Console.Write(vector[i] + " ");
            if (newLine == true)
                Console.WriteLine("\n");
        }
        /// <summary>
        /// 显示分类后的数据
        /// </summary>
        /// <param name="data">需要显示的数据</param>
        /// <param name="clustering">分组结果</param>
        /// <param name="numClusters">总组数</param>
        /// <param name="decimals">保留的小数位数</param>
        static void ShowClustered(double[][] data, int[] clustering, int numClusters, int decimals)
        {
            for (int k = 0; k < numClusters; ++k)
            {
                Console.WriteLine("===================");
                for (int i = 0; i < data.Length; ++i)
                {
                    if(clustering[i] == k)
                    {
                        Console.Write(i.ToString().PadLeft(3) + " ");
                        for (int j = 0; j < data[i].Length; ++j)
                        {
                            double v = data[i][j];
                            Console.Write(v.ToString("F" + decimals) + " ");
                        }
                        Console.WriteLine("");
                    }
                }
                Console.WriteLine("===================");
            }
        }
    }
}
