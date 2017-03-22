using System;

namespace NeuralClassification
{
    class NeuralProgram
    {
        /// <summary>
        /// PSO神经网络
        /// </summary>
        /// <param name="args">系统变量</param>
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin neural network demo\n");
            Console.WriteLine("Goal is to predict species from color, petal length, width \n");
            Console.WriteLine("Raw data looks like: \n");
            Console.WriteLine("blue, 1.4, 0.3, setosa");
            Console.WriteLine("pink, 4.9, 1.5, versicolor");
            Console.WriteLine("teal, 5.6, 1.8, virginica \n");
            double[][] trainData = new double[24][];
            trainData[0] = new double[] { 1, 0, 1.4, 0.3, 1, 0, 0 };
            trainData[1] = new double[] { 0, 1, 4.9, 1.5, 0, 1, 0 };
            trainData[2] = new double[] { -1, -1, 5.6, 1.8, 0, 0, 1 };
            trainData[3] = new double[] { -1, -1, 6.1, 2.5, 0, 0, 1 };
            trainData[4] = new double[] { 1, 0, 1.3, 0.2, 1, 0, 0 };
            trainData[5] = new double[] { 0, 1, 1.4, 0.2, 1, 0, 0 };
            trainData[6] = new double[] { 1, 0, 6.6, 2.1, 0, 0, 1 };
            trainData[7] = new double[] { 0, 1, 3.3, 1.0, 0, 1, 0 };
            trainData[8] = new double[] { -1, -1, 1.7, 0.4, 1, 0, 0 };
            trainData[9] = new double[] { 0, 1, 1.5, 0.1, 0, 1, 1 };
            trainData[10] = new double[] { 0, 1, 1.4, 0.2, 1, 0, 0 };
            trainData[11] = new double[] { 0, 1, 4.5, 1.5, 0, 1, 0 };
            trainData[12] = new double[] { 1, 0, 1.4, 0.2, 1, 0, 0 };
            trainData[13] = new double[] { -1, -1, 5.1, 1.9, 0, 0, 1 };
            trainData[14] = new double[] { 1, 0, 6.0, 2.5, 0, 0, 1 };
            trainData[15] = new double[] { 1, 0, 3.9, 1.4, 0, 1, 0 };
            trainData[16] = new double[] { 0, 1, 4.7, 1.4, 0, 1, 0 };
            trainData[17] = new double[] { -1, -1, 4.6, 1.5, 0, 1, 0 };
            trainData[18] = new double[] { -1, -1, 4.5, 1.7, 0, 0, 1 };
            trainData[19] = new double[] { 0, 1, 4.5, 1.3, 0, 1, 0 };
            trainData[20] = new double[] { 1, 0, 1.5, 0.2, 1, 0, 0 };
            trainData[21] = new double[] { 0, 1, 5.8, 2.2, 0, 0, 1 };
            trainData[22] = new double[] { 0, 1, 4.0, 1.3, 0, 1, 0 };
            trainData[23] = new double[] { -1, -1, 5.8, 1.8, 0, 0, 1 };
            double[][] testData = new double[6][];
            testData[0] = new double[] { 1, 0, 1.5, 0.2, 1, 0, 0 };
            testData[1] = new double[] { -1, -1, 5.9, 2.1, 0, 0, 1 };
            testData[2] = new double[] { 0, 1, 1.4, 0.2, 1, 0, 0 };
            testData[3] = new double[] { 0, 1, 4.7, 1.6, 0, 1, 0 };
            testData[4] = new double[] { 1, 0, 4.6, 1.3, 0, 1, 0 };
            testData[5] = new double[] { 1, 0, 6.3, 1.8, 0, 0, 1 };
            Console.WriteLine("Encoded training data is: \n");
            ShowData(trainData, 5, 1, true);
            Console.WriteLine("Encoded test data is: \n");
            ShowData(testData, 2, 1, true);
            Console.WriteLine("\nCreating a 4-input, 6-hidden, 3-output neural network");
            Console.WriteLine("Using tanh and softmax activations \n");
            int numInput = 4;
            int numHidden = 6;
            int numOutput = 3;
            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);
            int numParticles = 12;
            int maxEpochs = 500;
            Console.WriteLine("Setting numParticles = " + numParticles);
            Console.WriteLine("Setting maxEpochs = " + maxEpochs);
            Console.WriteLine("\nBeginning training using Particle Swarm Optimization");
            double[] bestWeights = nn.Train(trainData, numParticles, maxEpochs);
            Console.WriteLine("Training complete \n");
            Console.WriteLine("Final neural network weights and bias values:");
            ShowVector(bestWeights, 10, 3, true);
            nn.SetWeights(bestWeights);
            double trainAcc = nn.Accuracy(trainData);
            Console.WriteLine("\nAccuracy on training data = " + trainAcc.ToString("F4"));
            double testAcc = nn.Accuracy(testData);
            Console.WriteLine("Accuracy on test data = " + testAcc.ToString("F4"));
            Console.WriteLine("\nEnd neural network demo\n");
            Console.ReadLine();
        }
        /// <summary>
        /// 显示数据
        /// </summary>
        /// <param name="vector">一维数组</param>
        /// <param name="valsPerRow">一行的元素个数</param>
        /// <param name="decimals">小数位数</param>
        /// <param name="newLine">是否换新行</param>
        static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i % valsPerRow == 0) Console.WriteLine("");
                Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
            }
            if (newLine == true) Console.WriteLine("");
        }
        /// <summary>
        /// 显示数据
        /// </summary>
        /// <param name="data">二维数组</param>
        /// <param name="numRows">行数</param>
        /// <param name="decimals">小数位数</param>
        /// <param name="indices">是否是目录</param>
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
    }
}
