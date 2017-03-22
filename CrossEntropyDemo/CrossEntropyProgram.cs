using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CrossEntropyDemo
{
    class CrossEntropyProgram
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin cross entropy error neural network demo");
            Console.WriteLine("\nData is the famous Iris flower set.");
            Console.WriteLine("Input is sepal length, width, petal length, width");
            Console.WriteLine("Class to predict is species");
            Console.WriteLine("setosa = 0 0 1, versicolor = 0 1 0, virginica = 1 0 0 ");

            Console.WriteLine("\nRaw data has 150 total items:\n");
            Console.WriteLine(" 5.1, 3.5, 1.4, 0.2, Iris setosa");
            Console.WriteLine(" 7.0, 3.2, 4.7, 1.4, Iris versicolor");
            Console.WriteLine(" 6.3, 3.3, 6.0, 2.5, Iris virginica");
            Console.WriteLine(" ......\n");

            Console.WriteLine("Loading 80-20% training-test data");

            double[][] trainData = new double[120][];
            trainData[0] = new double[] { 6.0, 3.4, 4.5, 1.6, 0, 1, 0 };
            trainData[1] = new double[] { 6.7, 2.5, 5.8, 1.8, 1, 0, 0 };
            trainData[2] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
            trainData[3] = new double[] { 7.7, 2.8, 6.7, 2.0, 1, 0, 0 };
            trainData[4] = new double[] { 6.4, 3.2, 5.3, 2.3, 1, 0, 0 };
            trainData[5] = new double[] { 6.7, 3.1, 4.7, 1.5, 0, 1, 0 };
            trainData[6] = new double[] { 6.3, 3.4, 5.6, 2.4, 1, 0, 0 };
            trainData[7] = new double[] { 5.9, 3.2, 4.8, 1.8, 0, 1, 0 };
            trainData[8] = new double[] { 6.3, 2.5, 5.0, 1.9, 1, 0, 0 };
            trainData[9] = new double[] { 4.6, 3.2, 1.4, 0.2, 0, 0, 1 };
            trainData[10] = new double[] { 7.0, 3.2, 4.7, 1.4, 0, 1, 0 };
            trainData[11] = new double[] { 6.6, 3.0, 4.4, 1.4, 0, 1, 0 };
            trainData[12] = new double[] { 5.7, 2.8, 4.1, 1.3, 0, 1, 0 };
            trainData[13] = new double[] { 6.7, 3.0, 5.0, 1.7, 0, 1, 0 };
            trainData[14] = new double[] { 6.5, 3.0, 5.2, 2.0, 1, 0, 0 };
            trainData[15] = new double[] { 5.1, 3.8, 1.5, 0.3, 0, 0, 1 };
            trainData[16] = new double[] { 7.9, 3.8, 6.4, 2.0, 1, 0, 0 };
            trainData[17] = new double[] { 5.9, 3.0, 5.1, 1.8, 1, 0, 0 };
            trainData[18] = new double[] { 7.3, 2.9, 6.3, 1.8, 1, 0, 0 };
            trainData[19] = new double[] { 5.0, 2.0, 3.5, 1.0, 0, 1, 0 };
            trainData[20] = new double[] { 6.2, 2.8, 4.8, 1.8, 1, 0, 0 };
            trainData[21] = new double[] { 7.4, 2.8, 6.1, 1.9, 1, 0, 0 };
            trainData[22] = new double[] { 6.2, 3.4, 5.4, 2.3, 1, 0, 0 };
            trainData[23] = new double[] { 5.2, 3.5, 1.5, 0.2, 0, 0, 1 };
            trainData[24] = new double[] { 6.8, 3.0, 5.5, 2.1, 1, 0, 0 };
            trainData[25] = new double[] { 5.5, 2.6, 4.4, 1.2, 0, 1, 0 };
            trainData[26] = new double[] { 6.9, 3.1, 5.1, 2.3, 1, 0, 0 };
            trainData[27] = new double[] { 6.4, 2.7, 5.3, 1.9, 1, 0, 0 };
            trainData[28] = new double[] { 5.6, 2.7, 4.2, 1.3, 0, 1, 0 };
            trainData[29] = new double[] { 4.4, 3.0, 1.3, 0.2, 0, 0, 1 };
            trainData[30] = new double[] { 6.9, 3.1, 4.9, 1.5, 0, 1, 0 };
            trainData[31] = new double[] { 5.4, 3.0, 4.5, 1.5, 0, 1, 0 };
            trainData[32] = new double[] { 5.8, 2.7, 4.1, 1.0, 0, 1, 0 };
            trainData[33] = new double[] { 4.6, 3.6, 1.0, 0.2, 0, 0, 1 };
            trainData[34] = new double[] { 5.1, 3.5, 1.4, 0.2, 0, 0, 1 };
            trainData[35] = new double[] { 4.9, 3.0, 1.4, 0.2, 0, 0, 1 };
            trainData[36] = new double[] { 5.1, 3.4, 1.5, 0.2, 0, 0, 1 };
            trainData[37] = new double[] { 5.5, 2.4, 3.8, 1.1, 0, 1, 0 };
            trainData[38] = new double[] { 6.8, 2.8, 4.8, 1.4, 0, 1, 0 };
            trainData[39] = new double[] { 6.7, 3.0, 5.2, 2.3, 1, 0, 0 };
            trainData[40] = new double[] { 5.7, 3.0, 4.2, 1.2, 0, 1, 0 };
            trainData[41] = new double[] { 6.0, 2.2, 5.0, 1.5, 1, 0, 0 };
            trainData[42] = new double[] { 6.5, 2.8, 4.6, 1.5, 0, 1, 0 };
            trainData[43] = new double[] { 6.3, 2.5, 4.9, 1.5, 0, 1, 0 };
            trainData[44] = new double[] { 6.7, 3.1, 5.6, 2.4, 1, 0, 0 };
            trainData[45] = new double[] { 6.4, 2.8, 5.6, 2.1, 1, 0, 0 };
            trainData[46] = new double[] { 5.5, 2.4, 3.7, 1.0, 0, 1, 0 };
            trainData[47] = new double[] { 5.2, 3.4, 1.4, 0.2, 0, 0, 1 };
            trainData[48] = new double[] { 6.0, 2.2, 4.0, 1.0, 0, 1, 0 };
            trainData[49] = new double[] { 6.1, 2.8, 4.0, 1.3, 0, 1, 0 };
            trainData[50] = new double[] { 6.1, 3.0, 4.6, 1.4, 0, 1, 0 };
            trainData[51] = new double[] { 5.0, 3.2, 1.2, 0.2, 0, 0, 1 };
            trainData[52] = new double[] { 4.8, 3.4, 1.9, 0.2, 0, 0, 1 };
            trainData[53] = new double[] { 6.3, 3.3, 6.0, 2.5, 1, 0, 0 };
            trainData[54] = new double[] { 5.0, 3.5, 1.6, 0.6, 0, 0, 1 };
            trainData[55] = new double[] { 6.0, 3.0, 4.8, 1.8, 1, 0, 0 };
            trainData[56] = new double[] { 6.3, 2.8, 5.1, 1.5, 1, 0, 0 };
            trainData[57] = new double[] { 7.2, 3.2, 6.0, 1.8, 1, 0, 0 };
            trainData[58] = new double[] { 4.6, 3.4, 1.4, 0.3, 0, 0, 1 };
            trainData[59] = new double[] { 6.9, 3.2, 5.7, 2.3, 1, 0, 0 };
            trainData[60] = new double[] { 6.5, 3.0, 5.5, 1.8, 1, 0, 0 };
            trainData[61] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
            trainData[62] = new double[] { 5.1, 3.8, 1.9, 0.4, 0, 0, 1 };
            trainData[63] = new double[] { 5.3, 3.7, 1.5, 0.2, 0, 0, 1 };
            trainData[64] = new double[] { 5.0, 3.3, 1.4, 0.2, 0, 0, 1 };
            trainData[65] = new double[] { 6.6, 2.9, 4.6, 1.3, 0, 1, 0 };
            trainData[66] = new double[] { 5.9, 3.0, 4.2, 1.5, 0, 1, 0 };
            trainData[67] = new double[] { 5.2, 2.7, 3.9, 1.4, 0, 1, 0 };
            trainData[68] = new double[] { 7.7, 3.0, 6.1, 2.3, 1, 0, 0 };
            trainData[69] = new double[] { 5.4, 3.9, 1.7, 0.4, 0, 0, 1 };
            trainData[70] = new double[] { 5.1, 3.5, 1.4, 0.3, 0, 0, 1 };
            trainData[71] = new double[] { 7.2, 3.6, 6.1, 2.5, 1, 0, 0 };
            trainData[72] = new double[] { 6.4, 3.2, 4.5, 1.5, 0, 1, 0 };
            trainData[73] = new double[] { 4.8, 3.0, 1.4, 0.3, 0, 0, 1 };
            trainData[74] = new double[] { 6.2, 2.2, 4.5, 1.5, 0, 1, 0 };
            trainData[75] = new double[] { 5.8, 2.7, 3.9, 1.2, 0, 1, 0 };
            trainData[76] = new double[] { 7.2, 3.0, 5.8, 1.6, 1, 0, 0 };
            trainData[77] = new double[] { 4.7, 3.2, 1.6, 0.2, 0, 0, 1 };
            trainData[78] = new double[] { 6.3, 2.3, 4.4, 1.3, 0, 1, 0 };
            trainData[79] = new double[] { 6.8, 3.2, 5.9, 2.3, 1, 0, 0 };
            trainData[80] = new double[] { 5.0, 2.3, 3.3, 1.0, 0, 1, 0 };
            trainData[81] = new double[] { 5.7, 2.5, 5.0, 2.0, 1, 0, 0 };
            trainData[82] = new double[] { 7.7, 2.6, 6.9, 2.3, 1, 0, 0 };
            trainData[83] = new double[] { 6.5, 3.0, 5.8, 2.2, 1, 0, 0 };
            trainData[84] = new double[] { 6.1, 2.8, 4.7, 1.2, 0, 1, 0 };
            trainData[85] = new double[] { 4.7, 3.2, 1.3, 0.2, 0, 0, 1 };
            trainData[86] = new double[] { 6.9, 3.1, 5.4, 2.1, 1, 0, 0 };
            trainData[87] = new double[] { 6.4, 3.1, 5.5, 1.8, 1, 0, 0 };
            trainData[88] = new double[] { 6.0, 2.9, 4.5, 1.5, 0, 1, 0 };
            trainData[89] = new double[] { 6.4, 2.9, 4.3, 1.3, 0, 1, 0 };
            trainData[90] = new double[] { 4.4, 2.9, 1.4, 0.2, 0, 0, 1 };
            trainData[91] = new double[] { 5.0, 3.6, 1.4, 0.2, 0, 0, 1 };
            trainData[92] = new double[] { 4.4, 3.2, 1.3, 0.2, 0, 0, 1 };
            trainData[93] = new double[] { 5.1, 3.7, 1.5, 0.4, 0, 0, 1 };
            trainData[94] = new double[] { 4.8, 3.1, 1.6, 0.2, 0, 0, 1 };
            trainData[95] = new double[] { 6.5, 3.2, 5.1, 2.0, 1, 0, 0 };
            trainData[96] = new double[] { 6.1, 2.9, 4.7, 1.4, 0, 1, 0 };
            trainData[97] = new double[] { 5.4, 3.7, 1.5, 0.2, 0, 0, 1 };
            trainData[98] = new double[] { 5.7, 3.8, 1.7, 0.3, 0, 0, 1 };
            trainData[99] = new double[] { 7.1, 3.0, 5.9, 2.1, 1, 0, 0 };
            trainData[100] = new double[] { 5.4, 3.9, 1.3, 0.4, 0, 0, 1 };
            trainData[101] = new double[] { 6.1, 2.6, 5.6, 1.4, 1, 0, 0 };
            trainData[102] = new double[] { 6.4, 2.8, 5.6, 2.2, 1, 0, 0 };
            trainData[103] = new double[] { 5.0, 3.0, 1.6, 0.2, 0, 0, 1 };
            trainData[104] = new double[] { 5.8, 2.8, 5.1, 2.4, 1, 0, 0 };
            trainData[105] = new double[] { 6.3, 2.9, 5.6, 1.8, 1, 0, 0 };
            trainData[106] = new double[] { 6.2, 2.9, 4.3, 1.3, 0, 1, 0 };
            trainData[107] = new double[] { 5.5, 3.5, 1.3, 0.2, 0, 0, 1 };
            trainData[108] = new double[] { 6.7, 3.1, 4.4, 1.4, 0, 1, 0 };
            trainData[109] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
            trainData[110] = new double[] { 4.6, 3.1, 1.5, 0.2, 0, 0, 1 };
            trainData[111] = new double[] { 5.0, 3.5, 1.3, 0.3, 0, 0, 1 };
            trainData[112] = new double[] { 5.5, 2.5, 4.0, 1.3, 0, 1, 0 };
            trainData[113] = new double[] { 5.5, 4.2, 1.4, 0.2, 0, 0, 1 };
            trainData[114] = new double[] { 5.5, 2.3, 4.0, 1.3, 0, 1, 0 };
            trainData[115] = new double[] { 5.2, 4.1, 1.5, 0.1, 0, 0, 1 };
            trainData[116] = new double[] { 5.6, 2.5, 3.9, 1.1, 0, 1, 0 };
            trainData[117] = new double[] { 5.6, 2.9, 3.6, 1.3, 0, 1, 0 };
            trainData[118] = new double[] { 4.9, 2.4, 3.3, 1.0, 0, 1, 0 };
            trainData[119] = new double[] { 5.7, 2.8, 4.5, 1.3, 0, 1, 0 };

            double[][] testData = new double[30][];
            testData[0] = new double[] { 6.0, 2.7, 5.1, 1.6, 0, 1, 0 };
            testData[1] = new double[] { 5.1, 3.3, 1.7, 0.5, 0, 0, 1 };
            testData[2] = new double[] { 6.7, 3.3, 5.7, 2.1, 1, 0, 0 };
            testData[3] = new double[] { 5.1, 2.5, 3.0, 1.1, 0, 1, 0 };
            testData[4] = new double[] { 5.6, 2.8, 4.9, 2.0, 1, 0, 0 };
            testData[5] = new double[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
            testData[6] = new double[] { 5.0, 3.4, 1.5, 0.2, 0, 0, 1 };
            testData[7] = new double[] { 5.4, 3.4, 1.7, 0.2, 0, 0, 1 };
            testData[8] = new double[] { 4.9, 2.5, 4.5, 1.7, 1, 0, 0 };
            testData[9] = new double[] { 5.7, 4.4, 1.5, 0.4, 0, 0, 1 };
            testData[10] = new double[] { 7.7, 3.8, 6.7, 2.2, 1, 0, 0 };
            testData[11] = new double[] { 5.7, 2.9, 4.2, 1.3, 0, 1, 0 };
            testData[12] = new double[] { 5.0, 3.4, 1.6, 0.4, 0, 0, 1 };
            testData[13] = new double[] { 6.3, 3.3, 4.7, 1.6, 0, 1, 0 };
            testData[14] = new double[] { 4.5, 2.3, 1.3, 0.3, 0, 0, 1 };
            testData[15] = new double[] { 4.8, 3.4, 1.6, 0.2, 0, 0, 1 };
            testData[16] = new double[] { 5.8, 4.0, 1.2, 0.2, 0, 0, 1 };
            testData[17] = new double[] { 6.7, 3.3, 5.7, 2.5, 1, 0, 0 };
            testData[18] = new double[] { 4.3, 3.0, 1.1, 0.1, 0, 0, 1 };
            testData[19] = new double[] { 5.4, 3.4, 1.5, 0.4, 0, 0, 1 };
            testData[20] = new double[] { 5.6, 3.0, 4.1, 1.3, 0, 1, 0 };
            testData[21] = new double[] { 6.1, 3.0, 4.9, 1.8, 1, 0, 0 };
            testData[22] = new double[] { 5.7, 2.6, 3.5, 1.0, 0, 1, 0 };
            testData[23] = new double[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
            testData[24] = new double[] { 5.6, 3.0, 4.5, 1.5, 0, 1, 0 };
            testData[25] = new double[] { 4.8, 3.0, 1.4, 0.1, 0, 0, 1 };
            testData[26] = new double[] { 5.1, 3.8, 1.6, 0.2, 0, 0, 1 };
            testData[27] = new double[] { 7.6, 3.0, 6.6, 2.1, 1, 0, 0 };
            testData[28] = new double[] { 6.3, 2.7, 4.9, 1.8, 1, 0, 0 };
            testData[29] = new double[] { 5.8, 2.6, 4.0, 1.2, 0, 1, 0 };

            Console.WriteLine("\nFirst 5 rows of training data:");
            ShowMatrix(trainData, 5, 1, true);
            Console.WriteLine("First 3 rows of test data:");
            ShowMatrix(testData, 3, 1, true);

            Console.WriteLine("\nCreating a 4-input, 7-hidden, 3-output neural network");
            Console.WriteLine("Using tanh (hidden) and softmax (output) activations");
            const int numInput = 4;
            const int numHidden = 7;
            const int numOutput = 3;
            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);

            int maxEpochs = 2000;
            double learnRate = 0.05;
            Console.WriteLine("Setting maxEpochs = 2000, learnRate = 0.05");
            Console.WriteLine("No momentum or weight decay");
            Console.WriteLine("Training has hard-coded cross entropy error < 0.0250 stopping condition");

            Console.WriteLine("\nBeginning training using (incremental) back-propagation\n");
            nn.Train(trainData, maxEpochs, learnRate);
            Console.WriteLine("Training complete");

            double[] weights = nn.GetWeights();
            Console.WriteLine("Final neural network weights and bias values:");
            ShowVector(weights, 10, 3, true);

            double trainAcc = nn.Accuracy(trainData);
            Console.WriteLine("\nAccuracy on training data = " + trainAcc.ToString("F4"));

            double testAcc = nn.Accuracy(testData);
            Console.WriteLine("\nAccuracy on test data = " + testAcc.ToString("F4"));

            Console.WriteLine("\nEnd cross entropy demo\n");
            Console.ReadLine();

        }

        static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i % valsPerRow == 0) Console.WriteLine("");
                Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
            }
            if (newLine == true) Console.WriteLine("");
        }

        static void ShowMatrix(double[][] matrix, int numRows, int decimals, bool newLine)
        {
            for (int i = 0; i < numRows; ++i)
            {
                Console.Write(i.ToString().PadLeft(3) + ": ");
                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    if (matrix[i][j] >= 0.0) Console.Write(" "); else Console.Write("-"); ;
                    Console.Write(Math.Abs(matrix[i][j]).ToString("F" + decimals) + " ");
                }
                Console.WriteLine("");
            }
            if (newLine == true) Console.WriteLine("");
        }
    }
}
