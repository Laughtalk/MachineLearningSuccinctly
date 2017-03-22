using System;

namespace RBFNetwork
{
    class RBFProgram
    {
        /// <summary>
        /// RBF网络
        /// </summary>
        /// <param name="args">系统变量</param>
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin radial basis function (RBF) network training demo\n");
            Console.WriteLine("Goal is to train an RBF network on iris flower data to predict");
            Console.WriteLine("species from sepal length and width, and petal length and width.");

            // (0,0,1) = Iris setosa; (0,1,0) = Iris versicolor; (1,0,0) = Iris virginica
            // program assumes x-data precedes y-data

            double[][] allData = new double[30][];
            allData[0] = new double[] { -0.784, 1.255, -1.332, -1.306, 0, 0, 1 };
            allData[1] = new double[] { -0.995, -0.109, -1.332, -1.306, 0, 0, 1 };
            allData[2] = new double[] { -1.206, 0.436, -1.386, -1.306, 0, 0, 1 };
            allData[3] = new double[] { -1.312, 0.164, -1.278, -1.306, 0, 0, 1 };
            allData[4] = new double[] { -0.890, 1.528, -1.332, -1.306, 0, 0, 1 };
            allData[5] = new double[] { -0.468, 2.346, -1.170, -1.048, 0, 0, 1 };
            allData[6] = new double[] { -1.312, 0.982, -1.332, -1.177, 0, 0, 1 };
            allData[7] = new double[] { -0.890, 0.982, -1.278, -1.306, 0, 0, 1 };
            allData[8] = new double[] { -1.523, -0.382, -1.332, -1.306, 0, 0, 1 };
            allData[9] = new double[] { -0.995, 0.164, -1.278, -1.435, 0, 0, 1 };

            allData[10] = new double[] { 1.220, 0.436, 0.452, 0.241, 0, 1, 0 };
            allData[11] = new double[] { 0.587, 0.436, 0.344, 0.370, 0, 1, 0 };
            allData[12] = new double[] { 1.115, 0.164, 0.560, 0.370, 0, 1, 0 };
            allData[13] = new double[] { -0.362, -2.019, 0.074, 0.112, 0, 1, 0 };
            allData[14] = new double[] { 0.693, -0.655, 0.398, 0.370, 0, 1, 0 };
            allData[15] = new double[] { -0.151, -0.655, 0.344, 0.112, 0, 1, 0 };
            allData[16] = new double[] { 0.482, 0.709, 0.452, 0.498, 0, 1, 0 };
            allData[17] = new double[] { -0.995, -1.746, -0.305, -0.275, 0, 1, 0 };
            allData[18] = new double[] { 0.798, -0.382, 0.398, 0.112, 0, 1, 0 };
            allData[19] = new double[] { -0.679, -0.927, 0.020, 0.241, 0, 1, 0 };

            allData[20] = new double[] { 0.482, 0.709, 1.155, 1.659, 1, 0, 0 };
            allData[21] = new double[] { -0.046, -0.927, 0.669, 0.885, 1, 0, 0 };
            allData[22] = new double[] { 1.326, -0.109, 1.101, 1.143, 1, 0, 0 };
            allData[23] = new double[] { 0.482, -0.382, 0.939, 0.756, 1, 0, 0 };
            allData[24] = new double[] { 0.693, -0.109, 1.047, 1.272, 1, 0, 0 };
            allData[25] = new double[] { 1.853, -0.109, 1.479, 1.143, 1, 0, 0 };
            allData[26] = new double[] { -0.995, -1.473, 0.344, 0.627, 1, 0, 0 };
            allData[27] = new double[] { 1.537, -0.382, 1.317, 0.756, 1, 0, 0 };
            allData[28] = new double[] { 0.904, -1.473, 1.047, 0.756, 1, 0, 0 };
            allData[29] = new double[] { 1.431, 1.528, 1.209, 1.659, 1, 0, 0 };

            Console.WriteLine("\nFirst four and last line of normalized, encoded input data:\n");
            Helpers.ShowMatrix(allData, 4, 3, true, true);

            Console.WriteLine("\nSplitting data into 80%-20% train and test sets");
            double[][] trainData = null;
            double[][] testData = null;
            int seed = 8; // gives a good demo
            GetTrainTest(allData, seed, out trainData, out testData); // 80-20 hold-out 

            Console.WriteLine("\nTraining data: \n");
            Helpers.ShowMatrix(trainData, trainData.Length, 3, true, false);
            Console.WriteLine("\nTest data:\n");
            Helpers.ShowMatrix(testData, testData.Length, 3, true, false);

            Console.WriteLine("\nCreating a 4-5-3 radial basis function network");
            int numInput = 4;
            int numHidden = 5;
            int numOutput = 3;
            RadialNetwork rn = new RadialNetwork(numInput, numHidden, numOutput);

            Console.WriteLine("\nBeginning RBF training\n");
            int maxIterations = 100; // max for PSO 
            double[] bestWeights = rn.Train(trainData, maxIterations);

            Console.WriteLine("\nEvaluating result RBF classification accuracy on the test data");
            rn.SetWeights(bestWeights);

            double acc = rn.Accuracy(testData);
            Console.WriteLine("Classification accuracy = " + acc.ToString("F4"));

            Console.WriteLine("\nEnd RBF network training demo\n");
            Console.ReadLine();
        }
        /// <summary>
        /// 把数据随机按照 8：2 的比例分为训练集和测试集
        /// </summary>
        /// <param name="allData"></param>
        /// <param name="seed"></param>
        /// <param name="trainData"></param>
        /// <param name="testData"></param>
        static void GetTrainTest(double[][] allData, int seed, out double[][] trainData, out double[][] testData)
        {
            // 80-20 hold-out validation
            int[] allIndices = new int[allData.Length];
            for (int i = 0; i < allIndices.Length; ++i)
                allIndices[i] = i;

            Random rnd = new Random(seed);
            for (int i = 0; i < allIndices.Length; ++i) // shuffle indices
            {
                int r = rnd.Next(i, allIndices.Length);
                int tmp = allIndices[r];
                allIndices[r] = allIndices[i];
                allIndices[i] = tmp;
            }

            int numTrain = (int)(0.80 * allData.Length);
            int numTest = allData.Length - numTrain;

            trainData = new double[numTrain][];
            testData = new double[numTest][];

            int j = 0;
            for (int i = 0; i < numTrain; ++i)
                trainData[i] = allData[allIndices[j++]];
            for (int i = 0; i < numTest; ++i)
                testData[i] = allData[allIndices[j++]];

        }
    }

    // the raw data used in the demo is a 3-item subset of Fisher's Iris data set.
    //

    //double[][] rawData = new double[30][]; // 30-item subset of Iris data set (10 each class)
    //rawData[0] = new double[] { 5.1, 3.5, 1.4, 0.2, 0, 0, 1 }; // sepal length, sepal width, petal length, petal width -> 
    //rawData[1] = new double[] { 4.9, 3.0, 1.4, 0.2, 0, 0, 1 }; // Iris setosa = 0 0 1, Iris versicolor = 0 1 0, Iris virginica = 1 0 0
    //rawData[2] = new double[] { 4.7, 3.2, 1.3, 0.2, 0, 0, 1 };
    //rawData[3] = new double[] { 4.6, 3.1, 1.5, 0.2, 0, 0, 1 };
    //rawData[4] = new double[] { 5.0, 3.6, 1.4, 0.2, 0, 0, 1 };
    //rawData[5] = new double[] { 5.4, 3.9, 1.7, 0.4, 0, 0, 1 };
    //rawData[6] = new double[] { 4.6, 3.4, 1.4, 0.3, 0, 0, 1 };
    //rawData[7] = new double[] { 5.0, 3.4, 1.5, 0.2, 0, 0, 1 };
    //rawData[8] = new double[] { 4.4, 2.9, 1.4, 0.2, 0, 0, 1 };
    //rawData[9] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };

    //rawData[10] = new double[] { 7.0, 3.2, 4.7, 1.4, 0, 1, 0 };
    //rawData[11] = new double[] { 6.4, 3.2, 4.5, 1.5, 0, 1, 0 };
    //rawData[12] = new double[] { 6.9, 3.1, 4.9, 1.5, 0, 1, 0 };
    //rawData[13] = new double[] { 5.5, 2.3, 4.0, 1.3, 0, 1, 0 };
    //rawData[14] = new double[] { 6.5, 2.8, 4.6, 1.5, 0, 1, 0 };
    //rawData[15] = new double[] { 5.7, 2.8, 4.5, 1.3, 0, 1, 0 };
    //rawData[16] = new double[] { 6.3, 3.3, 4.7, 1.6, 0, 1, 0 };
    //rawData[17] = new double[] { 4.9, 2.4, 3.3, 1.0, 0, 1, 0 };
    //rawData[18] = new double[] { 6.6, 2.9, 4.6, 1.3, 0, 1, 0 };
    //rawData[19] = new double[] { 5.2, 2.7, 3.9, 1.4, 0, 1, 0 };

    //rawData[20] = new double[] { 6.3, 3.3, 6.0, 2.5, 1, 0, 0 };
    //rawData[21] = new double[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
    //rawData[22] = new double[] { 7.1, 3.0, 5.9, 2.1, 1, 0, 0 };
    //rawData[23] = new double[] { 6.3, 2.9, 5.6, 1.8, 1, 0, 0 };
    //rawData[24] = new double[] { 6.5, 3.0, 5.8, 2.2, 1, 0, 0 };
    //rawData[25] = new double[] { 7.6, 3.0, 6.6, 2.1, 1, 0, 0 };
    //rawData[26] = new double[] { 4.9, 2.5, 4.5, 1.7, 1, 0, 0 };
    //rawData[27] = new double[] { 7.3, 2.9, 6.3, 1.8, 1, 0, 0 };
    //rawData[28] = new double[] { 6.7, 2.5, 5.8, 1.8, 1, 0, 0 };
    //rawData[29] = new double[] { 7.2, 3.6, 6.1, 2.5, 1, 0, 0 };

    //double[] originalMeans = new double[] { 5.843, 3.040, 3.863, 1.213 }; // used to normalize original data
    //double[] originalStdDevs = new double[] { 0.948, 0.367, 1.850, 0.776 };

    //
}
