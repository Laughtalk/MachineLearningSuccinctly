using System;

namespace LogisticRegression
{
    class LogisticClassifier
    {
        private int numFeatures; // number of independent variables aka features
        private double[] weights; // b0 = constant
        private Random rnd;
        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="numFeatures">变量数量</param>
        public LogisticClassifier(int numFeatures)
        {
            this.numFeatures = numFeatures; // number of features/predictors
            this.weights = new double[numFeatures + 1]; // [0] = b0 constant
        }
        /// <summary>
        /// 训练模型
        /// </summary>
        /// <param name="trainData">训练数据</param>
        /// <param name="maxEpochs">训练次数</param>
        /// <param name="seed">随机数种子</param>
        /// <returns>返回结果权值</returns>
        public double[] Train(double[][] trainData, int maxEpochs, int seed)
        {
            // sort 3 solutions (small error to large)
            // compute centroid
            // if expanded is better than worst replace
            // else if reflected is better than worst, replace
            // else if contracted is better than worst, replace
            // else if random is better than worst, replace
            // else shrink
            this.rnd = new Random(seed); // so we can implement restart if wanted
            Solution[] solutions = new Solution[3]; // best, worst, other
            // initialize to random values
            for (int i = 0; i < 3; ++i)
            {
                solutions[i] = new Solution(numFeatures);
                solutions[i].weights = RandomSolutionWts();
                solutions[i].error = Error(trainData, solutions[i].weights);
            }
            const int best = 0; // const is better
            const int other = 1;
            const int worst = 2;
            int epoch = 0;
            while (epoch < maxEpochs)
            {
                ++epoch;
                Array.Sort(solutions); // [0] = best, [1] = other, [2] = worst
                double[] bestWts = solutions[0].weights; // convenience only
                double[] otherWts = solutions[1].weights;
                double[] worstWts = solutions[2].weights;
                double[] centroidWts = CentroidWts(otherWts, bestWts); // an average

                double[] expandedWts = ExpandedWts(centroidWts, worstWts);
                double expandedError = Error(trainData, expandedWts);
                if (expandedError < solutions[worst].error) // expanded better than worst?
                {
                    Array.Copy(expandedWts, worstWts, numFeatures + 1); // replace worst
                    solutions[worst].error = expandedError;
                    continue;
                }

                double[] reflectedWts = ReflectedWts(centroidWts, worstWts);
                double reflectedError = Error(trainData, reflectedWts);
                if (reflectedError < solutions[worst].error) // relected better than worst?
                {
                    Array.Copy(reflectedWts, worstWts, numFeatures + 1);
                    solutions[worst].error = reflectedError;
                    continue;
                }

                double[] contractedWts = ContractedWts(centroidWts, worstWts);
                double contractedError = Error(trainData, contractedWts);
                if (contractedError < solutions[worst].error) // contracted better than worst?
                {
                    Array.Copy(contractedWts, worstWts, numFeatures + 1);
                    solutions[worst].error = contractedError;
                    continue;
                }
                double[] randomSolWts = RandomSolutionWts();
                double randomSolError = Error(trainData, randomSolWts);
                if (randomSolError < solutions[worst].error)
                {
                    Array.Copy(randomSolWts, worstWts, numFeatures + 1);
                    solutions[worst].error = randomSolError;
                    continue;
                }

                // couldn't find a replacement for worst so shrink
                // worst -> towards best 距离折半
                for (int j = 0; j < numFeatures + 1; ++j)
                    worstWts[j] = (worstWts[j] + bestWts[j]) / 2.0;
                solutions[worst].error = Error(trainData, worstWts);
                // 'other' -> towards best 距离折半
                for (int j = 0; j < numFeatures + 1; ++j)
                    otherWts[j] = (otherWts[j] + bestWts[j]) / 2.0;
                solutions[other].error = Error(trainData, otherWts);
            } // while
            // copy best weights found, return by reference
            Array.Copy(solutions[best].weights, this.weights, this.numFeatures + 1);
            return this.weights;
        }
        /// <summary>
        /// 计算best和other的中值
        /// </summary>
        /// <param name="otherWts">other权值</param>
        /// <param name="bestWts">best权值</param>
        /// <returns>返回中值</returns>
        private double[] CentroidWts(double[] otherWts, double[] bestWts)
        {
            double[] result = new double[this.numFeatures + 1];
            for (int i = 0; i < result.Length; ++i)
                result[i] = (otherWts[i] + bestWts[i]) / 2.0;
            return result;
        }
        /// <summary>
        /// 三角形中线延长gama倍
        /// </summary>
        /// <param name="centroidWts">best和other的中点</param>
        /// <param name="worstWts">worst权值</param>
        /// <returns>返回Expanded权值</returns>
        private double[] ExpandedWts(double[] centroidWts, double[] worstWts)
        {
            double gamma = 2.0; // how far from centroid
            double[] result = new double[this.numFeatures + 1];
            for (int i = 0; i < result.Length; ++i)
                result[i] = centroidWts[i] + (gamma * (centroidWts[i] - worstWts[i]));
            return result;
        }
        /// <summary>
        /// 三角形中线延长alpha倍
        /// </summary>
        /// <param name="centroidWts">best和other的中点</param>
        /// <param name="worstWts">worst权值</param>
        /// <returns>返回Reflected权值</returns>
        private double[] ReflectedWts(double[] centroidWts, double[] worstWts)
        {
            double alpha = 1.0; // how far from centroid
            double[] result = new double[this.numFeatures + 1];
            for (int i = 0; i < result.Length; ++i)
                result[i] = centroidWts[i] + (alpha * (centroidWts[i] - worstWts[i]));
            return result;
        }
        /// <summary>
        /// 三角形中线的中点
        /// </summary>
        /// <param name="centroidWts">best和other的中点</param>
        /// <param name="worstWts">worst权值</param>
        /// <returns>返回Contracted权值</returns>
        private double[] ContractedWts(double[] centroidWts, double[] worstWts)
        {
            double rho = -0.5;
            double[] result = new double[this.numFeatures + 1];
            for (int i = 0; i < result.Length; ++i)
                result[i] = centroidWts[i] + (rho * (centroidWts[i] - worstWts[i]));
            return result;
        }
        /// <summary>
        /// 随机生成一组解
        /// </summary>
        /// <returns>返回随机解</returns>
        private double[] RandomSolutionWts()
        {
            double[] result = new double[this.numFeatures + 1];
            double lo = -10.0; //因为数据都已经规范化了，所以大部分权值会在-10到10之间。
            double hi = 10.0;
            for (int i = 0; i < result.Length; ++i)
                result[i] = (hi - lo) * rnd.NextDouble() + lo;
            return result;
        }
        /// <summary>
        /// 计算错误率
        /// </summary>
        /// <param name="trainData">训练数据</param>
        /// <param name="weights">权值</param>
        /// <returns>返回错误率</returns>
        private double Error(double[][] trainData, double[] weights)
        {
            // mean squared error using supplied weights
            int yIndex = trainData[0].Length - 1; // y-value (0/1) is last column
            double sumSquaredError = 0.0;
            for (int i = 0; i < trainData.Length; ++i) // each data
            {
                double computed = ComputeOutput(trainData[i], weights);
                double desired = trainData[i][yIndex]; // ex: 0.0 or 1.0
                sumSquaredError += (computed - desired) * (computed - desired);
            }
            return sumSquaredError / trainData.Length;
        }
        /// <summary>
        /// 计算给定数据和权值下Y的输出
        /// </summary>
        /// <param name="dataItem">数据</param>
        /// <param name="weights">权值</param>
        /// <returns>返回doubleY值（0.0~1.0）</returns>
        public double ComputeOutput(double[] dataItem, double[] weights)
        {
            double z = 0.0;
            z += weights[0]; // the b0 constant
            for (int i = 0; i < weights.Length - 1; ++i) // data might include Y
                z += (weights[i + 1] * dataItem[i]); // skip first weight
            return 1.0 / (1.0 + Math.Exp(-z)); // Y = 1.0 / (1.0 + e^(-Z) )
        }
        /// <summary>
        /// 给定数据和权值，先计算Y值，然后判断0 or 1.
        /// </summary>
        /// <param name="dataItem">数据</param>
        /// <param name="weights">权值</param>
        /// <returns>返回0 or 1</returns>
        public int ComputeDependent(double[] dataItem, double[] weights)
        {
            double sum = ComputeOutput(dataItem, weights);
            if (sum <= 0.5)
                return 0;
            else
                return 1;
        }
        /// <summary>
        /// 正确率
        /// </summary>
        /// <param name="trainData">测试数据</param>
        /// <param name="weights">测试权值</param>
        /// <returns>返回正确率</returns>
        public double Accuracy(double[][] trainData, double[] weights)
        {
            int numCorrect = 0;
            int numWrong = 0;
            int yIndex = trainData[0].Length - 1;
            for (int i = 0; i < trainData.Length; ++i)
            {
                double computed = ComputeDependent(trainData[i], weights); // implicit cast
                double desired = trainData[i][yIndex]; // 0.0 or 1.0
                if (computed == desired) // risky?
                    ++numCorrect;
                else
                    ++numWrong;
                //double closeness = 0.00000001;
                //if (Math.Abs(computed - desired) < closeness)
                // ++numCorrect;
                //else
                // ++numWrong;
            }
            return (numCorrect * 1.0) / (numWrong + numCorrect);
        }
        /// <summary>
        /// IComparable接口使Solution私有类拥有了自动排序的能力
        /// </summary>
        private class Solution : IComparable<Solution>
        {
            public double[] weights; // a potential solution
            public double error; // MSE of weights
            public Solution(int numFeatures)
            {
                this.weights = new double[numFeatures + 1]; // problem dim + constant
                this.error = 0.0;
            }
            public int CompareTo(Solution other) // low-to-high error
            {
                if (this.error < other.error)
                    return -1;
                else if (this.error > other.error)
                    return 1;
                else
                    return 0;
            }
        }
    }
}
