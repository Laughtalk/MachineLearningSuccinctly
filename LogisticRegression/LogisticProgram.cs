using System;

namespace LogisticRegression
{
    class LogisticProgram
    {
        /// <summary>
        /// 逻辑回归分类，适用于预测项为两个的情况。
        /// 例如：男女，死活，涨跌，输赢。
        /// </summary>
        /// <param name="args">系统变量</param>
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin Logistic Regression Binary Classification demo");
            Console.WriteLine("Goal is to predict death (0 = false, 1 = true)");
            double[][] data = new double[30][];
            data[0] = new double[] { 48, +1, 4.40, 0 };
            data[1] = new double[] { 60, -1, 7.89, 1 };
            data[2] = new double[] { 51, -1, 3.48, 0 };
            data[3] = new double[] { 66, -1, 8.41, 1 };
            data[4] = new double[] { 40, +1, 3.05, 0 };
            data[5] = new double[] { 44, +1, 4.56, 0 };
            data[6] = new double[] { 80, -1, 6.91, 1 };
            data[7] = new double[] { 52, -1, 5.69, 0 };
            data[8] = new double[] { 56, -1, 4.01, 0 };
            data[9] = new double[] { 55, -1, 4.48, 0 };
            data[10] = new double[] { 72, +1, 5.97, 0 };
            data[11] = new double[] { 57, -1, 6.71, 1 };
            data[12] = new double[] { 50, -1, 6.40, 0 };
            data[13] = new double[] { 80, -1, 6.67, 1 };
            data[14] = new double[] { 69, +1, 5.79, 0 };
            data[15] = new double[] { 39, -1, 5.42, 0 };
            data[16] = new double[] { 68, -1, 7.61, 1 };
            data[17] = new double[] { 47, +1, 3.24, 0 };
            data[18] = new double[] { 45, +1, 4.29, 0 };
            data[19] = new double[] { 79, +1, 7.44, 1 };
            data[20] = new double[] { 44, -1, 2.55, 0 };
            data[21] = new double[] { 52, +1, 3.71, 0 };
            data[22] = new double[] { 80, +1, 7.56, 1 };
            data[23] = new double[] { 76, -1, 7.80, 1 };
            data[24] = new double[] { 51, -1, 5.94, 0 };
            data[25] = new double[] { 46, +1, 5.52, 0 };
            data[26] = new double[] { 48, -1, 3.25, 0 };
            data[27] = new double[] { 58, +1, 4.71, 0 };
            data[28] = new double[] { 44, +1, 2.52, 0 };
            data[29] = new double[] { 68, -1, 8.38, 1 };
            Console.WriteLine("\nRaw data: \n");
            Console.WriteLine(" Age Sex Kidney Died");
            Console.WriteLine("=======================================");
            ShowData(data, 5, 2, true);
            Console.WriteLine("Normalizing age and kidney data");
            int[] columns = new int[] { 0, 2 };
            double[][] means = Normalize(data, columns); // normalize, save means and stdDevs
            Console.WriteLine("Done");
            Console.WriteLine("\nNormalized data: \n");
            ShowData(data, 5, 2, true);

            Console.WriteLine("Creating train (80%) and test (20%) matrices");
            double[][] trainData;
            double[][] testData;
            MakeTrainTest(data, 0, out trainData, out testData);
            Console.WriteLine("Done");
            Console.WriteLine("\nNormalized training data: \n");
            ShowData(trainData, 3, 2, true);
            //Console.WriteLine("\nFirst 3 rows and last row of normalized test data: \n");
            //ShowData(testData, 3, 2, true);

            int numFeatures = 3; // number of x-values (age, sex, kidney)
            Console.WriteLine("Creating LR binary classifier");
            LogisticClassifier lc = new LogisticClassifier(numFeatures);
            int maxEpochs = 100; // gives a representative demo
            Console.WriteLine("Setting maxEpochs = " + maxEpochs);
            Console.WriteLine("Starting training using simplex optimization");
            double[] bestWeights = lc.Train(trainData, maxEpochs, 33); // 33 = 'nice' demo
            Console.WriteLine("Training complete");
            Console.WriteLine("\nBest weights found:");
            ShowVector(bestWeights, 4, true);
            double trainAccuracy = lc.Accuracy(trainData, bestWeights);
            Console.WriteLine("Prediction accuracy on training data = " +
            trainAccuracy.ToString("F4"));
            double testAccuracy = lc.Accuracy(testData, bestWeights);
            Console.WriteLine("Prediction accuracy on test data = " +
            testAccuracy.ToString("F4"));
            //double[][] unknown = new double[1][];
            //unknown[0] = new double[] { 58.0, -1.0, 7.00 };
            //Normalize(unknown, columns, means);
            //int died = lc.ComputeDependent(unknown[0], bestWeights);
            //Console.WriteLine("Died = " + died);
            Console.WriteLine("\nEnd LR binary classification demo\n");
            Console.ReadLine();
        }
        /// <summary>
        /// 将二维数组中的指定列规范化（不需要提供均值和标准差）
        /// </summary>
        /// <param name="rawData">待规范化的double二维数组</param>
        /// <param name="columns">保存了需要规范化的列号</param>
        /// <returns>返回原数据的均值和标准差</returns>
        static double[][] Normalize(double[][] rawData, int[] columns)
        {
            // return means and sdtDevs of all columns for later use
            int numRows = rawData.Length;
            int numCols = rawData[0].Length;
            double[][] result = new double[2][]; // [0] = mean, [1] = stdDev
            for (int i = 0; i < 2; ++i)
                result[i] = new double[numCols];
            for (int c = 0; c < numCols; ++c)
            {
                // means of all cols
                double sum = 0.0;
                for (int r = 0; r < numRows; ++r)
                    sum += rawData[r][c];
                double mean = sum / numRows;
                result[0][c] = mean; // save
                // stdDevs of all cols
                double sumSquares = 0.0;
                for (int r = 0; r < numRows; ++r)
                    sumSquares += (rawData[r][c] - mean) * (rawData[r][c] - mean);
                double stdDev = Math.Sqrt(sumSquares / numRows);
                result[1][c] = stdDev;
            }
            // normalize
            for (int c = 0; c < columns.Length; ++c)
            {
                int j = columns[c]; // column to normalize
                double mean = result[0][j]; // mean of the col
                double stdDev = result[1][j];
                for (int i = 0; i < numRows; ++i)
                    rawData[i][j] = (rawData[i][j] - mean) / stdDev;
            }
            return result;
        }
        /// <summary>
        /// 将二维数组中的指定列规范化（需要提供均值和标准差）
        /// </summary>
        /// <param name="rawData">待规范化的double二维数组</param>
        /// <param name="columns">保存了需要规范化的列号</param>
        /// <param name="means">输入均值和标准差</param>
        static void Normalize(double[][] rawData, int[] columns, double[][] means)
        {
            // normalize columns using supplied means and standard devs
            int numRows = rawData.Length;
            for (int c = 0; c < columns.Length; ++c) // each specified col
            {
                int j = columns[c]; // column to normalize
                double mean = means[0][j];
                double stdDev = means[1][j];
                for (int i = 0; i < numRows; ++i) // each row
                    rawData[i][j] = (rawData[i][j] - mean) / stdDev;
            }
        }
        /// <summary>
        /// 用Fisher-Yates把数据分为训练集和测试集
        /// </summary>
        /// <param name="allData">全部数据</param>
        /// <param name="seed">随机数种子</param>
        /// <param name="trainData">返回训练集</param>
        /// <param name="testData">返回测试集</param>
        static void MakeTrainTest(double[][] allData, int seed, out double[][] trainData, out double[][] testData)
        {
            Random rnd = new Random(seed);
            int totRows = allData.Length;
            int numTrainRows = (int)(totRows * 0.80); // 80% hard-coded
            int numTestRows = totRows - numTrainRows;
            trainData = new double[numTrainRows][];
            testData = new double[numTestRows][];
            double[][] copy = new double[allData.Length][]; // ref copy of all data
            for (int i = 0; i < copy.Length; ++i)
                copy[i] = allData[i];
            for (int i = 0; i < copy.Length; ++i) // scramble order
            {
                int r = rnd.Next(i, copy.Length); // use Fisher-Yates
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }
            for (int i = 0; i < numTrainRows; ++i)
                trainData[i] = copy[i];
            for (int i = 0; i < numTestRows; ++i)
                testData[i] = copy[i + numTrainRows];
        }
        /// <summary>
        /// 显示二维double数据
        /// </summary>
        /// <param name="data">数据</param>
        /// <param name="numRows">行数</param>
        /// <param name="decimals">小数位数</param>
        /// <param name="indices">是否目录</param>
        static void ShowData(double[][] data, int numRows, int decimals, bool indices)
        {
            for (int i = 0; i < numRows; ++i)
            {
                if (indices == true)
                    Console.Write("[" + i.ToString().PadLeft(2) + "] ");
                for (int j = 0; j < data[i].Length; ++j)
                {
                    double v = data[i][j];
                    if (v >= 0.0)
                        Console.Write(" "); // '+'
                    Console.Write(v.ToString("F" + decimals) + " ");
                }
                Console.WriteLine("");
            }
            Console.WriteLine(". . .");
            int lastRow = data.Length - 1;
            if (indices == true)
                Console.Write("[" + lastRow.ToString().PadLeft(2) + "] ");
            for (int j = 0; j < data[lastRow].Length; ++j)
            {
                double v = data[lastRow][j];
                if (v >= 0.0)
                    Console.Write(" "); // '+'
                Console.Write(v.ToString("F" + decimals) + " ");
            }
            Console.WriteLine("\n");
        }
        /// <summary>
        /// 显示一维double数据
        /// </summary>
        /// <param name="vector">数据</param>
        /// <param name="decimals">小数位数</param>
        /// <param name="newLine">是否新行</param>
        static void ShowVector(double[] vector, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
                Console.Write(vector[i].ToString("F" + decimals) + " ");
            Console.WriteLine("");
            if (newLine == true)
                Console.WriteLine("");
        }
    }
}
