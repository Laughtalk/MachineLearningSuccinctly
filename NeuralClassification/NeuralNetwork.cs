using System;

namespace NeuralClassification
{
    class NeuralNetwork
    {
        private int numInput; // number of input nodes
        private int numHidden;
        private int numOutput;
        private double[] inputs;
        private double[][] ihWeights; // input-hidden
        private double[] hBiases;
        private double[] hOutputs;
        private double[][] hoWeights; // hidden-output
        private double[] oBiases;
        private double[] outputs;
        private Random rnd;
        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="numInput">输入层</param>
        /// <param name="numHidden">隐藏层</param>
        /// <param name="numOutput">输出层</param>
        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;
            this.inputs = new double[numInput];
            this.ihWeights = MakeMatrix(numInput, numHidden);
            this.hBiases = new double[numHidden];
            this.hOutputs = new double[numHidden];
            this.hoWeights = MakeMatrix(numHidden, numOutput);
            this.oBiases = new double[numOutput];
            this.outputs = new double[numOutput];
            this.rnd = new Random(0);
        }
        /// <summary>
        /// 生成一个double类型rows行cols列的矩阵
        /// </summary>
        private static double[][] MakeMatrix(int rows, int cols) // helper for ctor
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            return result;
        }
        /// <summary>
        /// 设置神经网络的连接权值
        /// </summary>
        /// <param name="weights">输入权值</param>
        public void SetWeights(double[] weights)
        {
            // copy weights and biases in weights[] array to i-h weights,
            // i-h biases, h-o weights, h-o biases
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) +
            numHidden + numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array length: ");
            int k = 0; // points into weights param
            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numHidden; ++j)
                    ihWeights[i][j] = weights[k++];
            for (int i = 0; i < numHidden; ++i)
                hBiases[i] = weights[k++];
            for (int i = 0; i < numHidden; ++i)
                for (int j = 0; j < numOutput; ++j)
                    hoWeights[i][j] = weights[k++];
            for (int i = 0; i < numOutput; ++i)
                oBiases[i] = weights[k++];
        }
        /// <summary>
        /// 返回神经网络的输出
        /// </summary>
        /// <param name="xValues"输入></param>
        /// <returns>输出</returns>
        public double[] ComputeOutputs(double[] xValues)
        {
            double[] hSums = new double[numHidden]; // hidden nodes sums scratch array
            double[] oSums = new double[numOutput]; // output nodes sums
            for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
                this.inputs[i] = xValues[i];
            for (int j = 0; j < numHidden; ++j) // compute i-h sum of weights * inputs
                for (int i = 0; i < numInput; ++i)
                    hSums[j] += this.inputs[i] * this.ihWeights[i][j]; // note +=
            for (int i = 0; i < numHidden; ++i) // add biases to input-to-hidden sums
                hSums[i] += this.hBiases[i];
            for (int i = 0; i < numHidden; ++i) // apply activation
                this.hOutputs[i] = HyperTan(hSums[i]); // hard-coded
            for (int j = 0; j < numOutput; ++j) // compute h-o sum of weights * hOutputs
                for (int i = 0; i < numHidden; ++i)
                    oSums[j] += hOutputs[i] * hoWeights[i][j];
            for (int i = 0; i < numOutput; ++i) // add biases to input-to-hidden sums
                oSums[i] += oBiases[i];
            double[] softOut = Softmax(oSums); // all outputs at once for efficiency
            Array.Copy(softOut, outputs, softOut.Length);
            double[] retResult = new double[numOutput];
            Array.Copy(this.outputs, retResult, retResult.Length);
            return retResult;
        }
        /// <summary>
        /// 计算tanh(x)
        /// </summary>
        private static double HyperTan(double x)
        {
            if (x < -20.0)
                return -1.0; // approximation is correct to 30 decimals
            else if (x > 20.0)
                return 1.0;
            else
                return Math.Tanh(x);
        }
        /// <summary>
        /// 使输入的值的和为1
        /// 因为答案的三个输出值只有一个为1其他为0
        /// </summary>
        /// <param name="oSums">输入</param>
        /// <returns>返回和为1的序列</returns>
        private static double[] Softmax(double[] oSums)
        {
            // does all output nodes at once so scale doesn't have to be re-computed each time
            // determine max output-sum
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
                if (oSums[i] > max) max = oSums[i];
            // determine scaling factor -- sum of exp(each val - max)
            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                scale += Math.Exp(oSums[i] - max);
            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i] - max) / scale;
            return result; // now scaled so that xi sum to 1.0
        }
        /// <summary>
        /// 训练函数
        /// </summary>
        /// <param name="trainData">训练数据</param>
        /// <param name="numParticles">粒子数量</param>
        /// <param name="maxEpochs">训练次数</param>
        /// <returns>返回权值</returns>
        public double[] Train(double[][] trainData, int numParticles, int maxEpochs)
        {
            int numWeights = (this.numInput * this.numHidden) + this.numHidden +
            (this.numHidden * this.numOutput) + this.numOutput;
            // use PSO to seek best weights
            int epoch = 0;
            double minX = -10.0; // for each weight. assumes data is normalized or 'nice'
            double maxX = 10.0;
            double w = 0.729; // inertia weight
            double c1 = 1.49445; // cognitive weight
            double c2 = 1.49445; // social weight
            double r1, r2; // cognitive and social randomizations
            Particle[] swarm = new Particle[numParticles];
            // best solution found by any particle in the swarm
            double[] bestGlobalPosition = new double[numWeights];
            double bestGlobalError = double.MaxValue; // smaller values better
            // initialize each Particle in the swarm with random positions and velocities
            double lo = 0.1 * minX;
            double hi = 0.1 * maxX;
            //给每个粒子随机位置和随机速度，并计算均方差，找出最佳位置
            for (int i = 0; i < swarm.Length; ++i)
            {
                double[] randomPosition = new double[numWeights];
                for (int j = 0; j < randomPosition.Length; ++j)
                    randomPosition[j] = (maxX - minX) * rnd.NextDouble() + minX;
                double error = MeanSquaredError(trainData, randomPosition);
                double[] randomVelocity = new double[numWeights];
                for (int j = 0; j < randomVelocity.Length; ++j)
                    randomVelocity[j] = (hi - lo) * rnd.NextDouble() + lo;
                swarm[i] = new Particle(randomPosition, error, randomVelocity, randomPosition, error);
                // does current Particle have global best position/solution?
                if (swarm[i].error < bestGlobalError)
                {
                    bestGlobalError = swarm[i].error;
                    swarm[i].position.CopyTo(bestGlobalPosition, 0);
                }
            }
            // main PSO algorithm
            int[] sequence = new int[numParticles]; // process particles in random order
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i; 
            while (epoch < maxEpochs)
            {
                double[] newVelocity = new double[numWeights]; // step 1
                double[] newPosition = new double[numWeights]; // step 2
                double newError; // step 3
                Shuffle(sequence); // move particles in random sequence
                for (int pi = 0; pi < swarm.Length; ++pi) // each Particle (index)
                {
                    int i = sequence[pi];
                    Particle currP = swarm[i]; // for coding convenience
                    // 1. compute new velocity
                    for (int j = 0; j < currP.velocity.Length; ++j) // each value of the velocity
                    {
                        r1 = rnd.NextDouble();
                        r2 = rnd.NextDouble();
                        // velocity depends on old velocity, best position of particle, and
                        // best position of any particle
                        newVelocity[j] = (w * currP.velocity[j]) +
                        (c1 * r1 * (currP.bestPosition[j] - currP.position[j])) +
                        (c2 * r2 * (bestGlobalPosition[j] - currP.position[j]));
                    }
                    newVelocity.CopyTo(currP.velocity, 0);
                    // 2. use new velocity to compute new position
                    for (int j = 0; j < currP.position.Length; ++j)
                    {
                        newPosition[j] = currP.position[j] + newVelocity[j];
                        if (newPosition[j] < minX) // keep in range
                            newPosition[j] = minX;
                        else if (newPosition[j] > maxX)
                            newPosition[j] = maxX;
                    }
                    newPosition.CopyTo(currP.position, 0);
                    // 3. compute error of new position
                    newError = MeanSquaredError(trainData, newPosition);
                    currP.error = newError;
                    if (newError < currP.bestError) // new particle best?
                    {
                        newPosition.CopyTo(currP.bestPosition, 0);
                        currP.bestError = newError;
                    }
                    if (newError < bestGlobalError) // new global best?
                    {
                        newPosition.CopyTo(bestGlobalPosition, 0);
                        bestGlobalError = newError;
                    }
                } // each Particle
                ++epoch;
            } // while
            SetWeights(bestGlobalPosition); // best position is a set of weights
            double[] retResult = new double[numWeights];
            Array.Copy(bestGlobalPosition, retResult, retResult.Length);
            return retResult;
        }
        /// <summary>
        /// 洗牌算法
        /// </summary>
        private void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int ri = rnd.Next(i, sequence.Length);
                int tmp = sequence[ri];
                sequence[ri] = sequence[i];
                sequence[i] = tmp;
            }
        }
        /// <summary>
        /// 计算给定权值和给定数据下的均方差
        /// </summary>
        /// <param name="trainData">训练数据</param>
        /// <param name="weights">权值</param>
        /// <returns>返回均方差</returns>
        private double MeanSquaredError(double[][] trainData, double[] weights)
        {
            this.SetWeights(weights); // copy the weights to evaluate in
            double[] xValues = new double[numInput]; // inputs
            double[] tValues = new double[numOutput]; // targets
            double sumSquaredError = 0.0;
            for (int i = 0; i < trainData.Length; ++i) // walk through each training item
            {
                // the following assumes data has all x-values first, followed by y-values!
                Array.Copy(trainData[i], xValues, numInput); // extract inputs
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // extract targets
                double[] yValues = this.ComputeOutputs(xValues);
                for (int j = 0; j < yValues.Length; ++j)
                    sumSquaredError += ((yValues[j] - tValues[j]) * (yValues[j] - tValues[j]));
            }
            return sumSquaredError / trainData.Length;
        }
        /// <summary>
        /// 计算给定数据的正确率
        /// </summary>
        /// <param name="testData">测试数据</param>
        /// <returns>返回正确率</returns>
        public double Accuracy(double[][] testData)
        {
            // percentage correct using winner-takes all
            int numCorrect = 0;
            int numWrong = 0;
            double[] xValues = new double[numInput]; // inputs
            double[] tValues = new double[numOutput]; // targets
            double[] yValues; // computed Y
            for (int i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, numInput); // parse test data
                Array.Copy(testData[i], numInput, tValues, 0, numOutput);
                yValues = this.ComputeOutputs(xValues);
                int maxIndex = IndexOfMax(yValues); // which cell in yValues has largest value?
                if (tValues[maxIndex] == 1.0) // ugly
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong);
        }
        /// <summary>
        /// 返回最大值的下标
        /// </summary>
        private static int IndexOfMax(double[] vector) // helper for Accuracy()
        {
            // index of largest value
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i];
                    bigIndex = i;
                }
            }
            return bigIndex;
        }
        /// <summary>
        /// 私有类，表示粒子
        /// </summary>
        private class Particle
        {
            public double[] position; // equivalent to NN weights
            public double error; // measure of fitness
            public double[] velocity;
            public double[] bestPosition; // best position found by this Particle
            public double bestError;
            public Particle(double[] position, double error, double[] velocity, double[] bestPosition, double bestError)
            {
                this.position = new double[position.Length];
                position.CopyTo(this.position, 0);
                this.error = error;
                this.velocity = new double[velocity.Length];
                velocity.CopyTo(this.velocity, 0);
                this.bestPosition = new double[bestPosition.Length];
                bestPosition.CopyTo(this.bestPosition, 0);
                this.bestError = bestError;
            }
        }
    }
}
