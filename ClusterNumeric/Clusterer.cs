using System;

namespace ClusterNumeric
{
    class Clusterer
    {
        private int numClusters; // number of clusters
        private int[] clustering; // index = a tuple, value = cluster ID
        private double[][] centroids; // mean (vector) of each cluster
        private Random rnd; // for initialization
        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="numClusters">分组组数</param>
        public Clusterer(int numClusters)
        {
            this.numClusters = numClusters;
            this.centroids = new double[numClusters][];
            this.rnd = new Random(0); // arbitrary seed
        }
        /// <summary>
        /// 对传入的数据进行分组
        /// 组数在构造类的时候设置了
        /// </summary>
        /// <param name="data">用于分类的数据</param>
        /// <returns>返回分组结果</returns>
        public int[] Cluster(double[][] data)
        {
            //给每个中心初始化
            int numTuples = data.Length;
            int numValues = data[0].Length;
            this.clustering = new int[numTuples];
            for (int k = 0; k < numClusters; ++k) // allocate each centroid
                this.centroids[k] = new double[numValues];

            InitRandom(data);
            Console.WriteLine("\nInitial random clustering:");
            for (int i = 0; i < clustering.Length; ++i)
                Console.Write(clustering[i] + " ");
            Console.WriteLine("\n");

            bool changed = true; // change in clustering?
            int maxCount = numTuples * 10; // sanity check
            int ct = 0;
            while (changed == true && ct <= maxCount)
            {
                ++ct; // k-means typically converges very quickly
                UpdateCentroids(data); // no effect if fail
                changed = UpdateClustering(data); // no effect if fail
            }
            int[] result = new int[numTuples];
            Array.Copy(this.clustering, result, clustering.Length);
            return result;
        }
        /// <summary>
        /// 对分组情况进行随机的初始化
        /// </summary>
        /// <param name="data">传入data只是为了使用其长度</param>
        private void InitRandom(double[][] data)
        {
            int numTuples = data.Length;
            int clusterID = 0;
            for (int i = 0; i < numTuples; ++i)
            {
                clustering[i] = clusterID++;
                if (clusterID == numClusters)
                    clusterID = 0;
            }
            //shuffle the values in the array (using Fisher-Yates)
            for (int i = 0; i < numTuples; ++i)
            {
                int r = rnd.Next(i, clustering.Length);
                int tmp = clustering[r];
                clustering[r] = clustering[i];
                clustering[i] = tmp;
            }
        }
        /// <summary>
        /// 更新每一个中心
        /// </summary>
        /// <param name="data">需要分类的数据</param>
        private void UpdateCentroids(double[][] data)
        {
            //每组分类的tuple个数
            int[] clusterCounts = new int[numClusters];
            for (int i = 0; i < data.Length; ++i)
                ++clusterCounts[clustering[i]];

            // zero-out this.centroids so it can be used as scratch 
            for (int k = 0; k < centroids.Length; ++k)
                for (int j = 0; j < centroids[k].Length; ++j)
                    centroids[k][j] = 0.0;

            // accumulate sum
            for (int i = 0; i < data.Length; ++i)
                for (int j = 0; j < data[i].Length; ++j)
                    centroids[clustering[i]][j] += data[i][j];

            // accumulate average
            for (int k = 0; k < centroids.Length; ++k)
                for (int j = 0; j < centroids[k].Length; ++j)
                    if (clusterCounts[k] >= 0)
                        centroids[k][j] /= clusterCounts[k];
        }
        /// <summary>
        /// 更新分组信息
        /// </summary>
        /// <param name="data">需要分组的数据</param>
        /// <returns>返回是否有元组发生了组别的变化</returns>
        private bool UpdateClustering(double[][] data)
        {
            // (re)assign each tuple to a cluster (closest centroid)
            // returns false if no tuple assignments change OR
            // if the reassignment would result in a clustering where
            // one or more clusters have no tuples.
            bool changed = false; // did any tuple change cluster?
            int[] newClustering = new int[clustering.Length]; // proposed result
            Array.Copy(clustering, newClustering, clustering.Length);
            double[] distances = new double[numClusters]; // from tuple to centroids
            for (int i = 0; i < data.Length; ++i) // walk through each tuple
            {
                for (int k = 0; k < numClusters; ++k)
                    distances[k] = Distance(data[i], centroids[k]);
                int newClusterID = MinIndex(distances); // find closest centroid
                if (newClusterID != newClustering[i])
                {
                    changed = true; // note a new clustering
                    newClustering[i] = newClusterID; // accept update
                }
            }
            if (changed == false)
                return false; // no change so bail
            // check proposed clustering cluster counts
            int[] clusterCounts = new int[numClusters];
            for (int i = 0; i < data.Length; ++i)
            {
                int clusterID = newClustering[i];
                ++clusterCounts[clusterID];
            }
            for (int k = 0; k < numClusters; ++k)
                if (clusterCounts[k] == 0)
                    return false; // bad clustering
            // alternative: place a random data item into empty cluster
            //for (int k = 0; k < numClusters; ++k)
            //{
            //    if (clusterCounts[k] == 0) // cluster k has no items
            //    {
            //        for (int t = 0; t < data.Length; ++t) // find a tuple to put into cluster k
            //        {
            //            int cid = newClustering[t]; // cluster of t
            //            int ct = clusterCounts[cid]; // how many items are there?
            //            if (ct >= 2) // t is in a cluster w/ 2 or more items
            //            {
            //                newClustering[t] = k; // place t into cluster k
            //                ++clusterCounts[k]; // k now has a data item
            //                --clusterCounts[cid]; // cluster that used to have t 
            //                break; // check next cluster
            //            }
            //        } // t
            //    } // cluster count of 0
            //} // k
            Array.Copy(newClustering, clustering, newClustering.Length); // update
            return true; // good clustering and at least one change
        }
        /// <summary>
        /// 几何距离
        /// </summary>
        /// <param name="tuple">数据点</param>
        /// <param name="centroid">该组的中心</param>
        /// <returns></returns>
        private static double Distance(double[] tuple, double[] centroid)
        {
            // Euclidean distance between two vectors for UpdateClustering()
            double sumSquaredDiffs = 0.0;
            for (int j = 0; j < tuple.Length; ++j)
                sumSquaredDiffs += (tuple[j] - centroid[j]) * (tuple[j] - centroid[j]);
            return Math.Sqrt(sumSquaredDiffs);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="distances"></param>
        /// <returns></returns>
        private static int MinIndex(double[] distances)
        {
            // helper for UpdateClustering() to find closest centroid
            int indexOfMin = 0;
            for (int k = 1; k < distances.Length; ++k)
            {
                if (distances[k] < distances[indexOfMin])
                {
                    indexOfMin = k;
                }
            }
            return indexOfMin;
        }
    }
}
