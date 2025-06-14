// --- НАЧАЛО ФИНАЛЬНОГО ОБЪЕДИНЕННОГО ФАЙЛА C# (ИСПРАВЛЕННАЯ ВЕРСИЯ 4 - HEADER FIX) ---

using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.CommandLine;
using System.Diagnostics;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace GemmaSharp
{
    public static class Quantization
    {
        // ИСПРАВЛЕНИЕ: enum QuantType теперь имеет базовый тип byte, чтобы занимать 1 байт, как в файле.
        public enum QuantType : byte
        {
            None,
            Q8_0,
        }
        
        public class QuantizedTensor
        {
            public ReadOnlyMemory<sbyte> Q { get; }
            public ReadOnlyMemory<float> S { get; }

            public QuantizedTensor(ReadOnlyMemory<sbyte> q, ReadOnlyMemory<float> s)
            {
                Q = q;
                S = s;
            }
        }
        
        public class MutableQuantizedTensor
        {
            public Memory<sbyte> Q { get; }
            public Memory<float> S { get; }

            public MutableQuantizedTensor(Memory<sbyte> q, Memory<float> s)
            {
                Q = q;
                S = s;
            }
        }
        
        public static void Dequantize(QuantizedTensor qx, Span<float> x, int n, uint gs)
        {
            if (gs == 0) return;

            var qSpan = qx.Q.Span;
            var sSpan = qx.S.Span;
            if (sSpan.IsEmpty) return;

            for (int i = 0; i < n; i++)
            {
                x[i] = qSpan[i] * sSpan[(int)(i / gs)];
            }
        }
        
        public static void Quantize(MutableQuantizedTensor qx, ReadOnlySpan<float> x, int n, uint gs)
        {
            if (gs == 0) return;

            var qSpan = qx.Q.Span;
            var sSpan = qx.S.Span;
            uint numGroups = (uint)n / gs;
            const float qMax = 127.0f;

            for (uint group = 0; group < numGroups; group++)
            {
                float wmax = 0.0f;
                int groupStart = (int)(group * gs);

                for (int i = 0; i < gs; i++)
                {
                    float val = Math.Abs(x[groupStart + i]);
                    if (val > wmax)
                    {
                        wmax = val;
                    }
                }

                float scale = wmax / qMax;
                sSpan[(int)group] = scale;
                
                if (scale == 0)
                {
                    qSpan.Slice(groupStart, (int)gs).Fill(0);
                    continue;
                }

                for (int i = 0; i < gs; i++)
                {
                    float quantValue = x[groupStart + i] / scale;
                    qSpan[groupStart + i] = (sbyte)Math.Clamp(Math.Round(quantValue), sbyte.MinValue, sbyte.MaxValue);
                }
            }
        }
    }

    public static class Functional
    {
        public static uint SliceToU32(ReadOnlySpan<byte> slice) => BinaryPrimitives.ReadUInt32LittleEndian(slice);
        public static float SliceToF32(ReadOnlySpan<byte> slice) => BitConverter.ToSingle(slice);

        public static uint RandomU32(ref ulong state)
        {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            return (uint)((state * 0x2545F4914F6CDD1DuL) >> 32);
        }

        public static float RandomF32(ref ulong state) => (RandomU32(ref state) >> 8) / 16777216.0f;

        public static void RmsNorm(Span<float> o, ReadOnlySpan<float> x, ReadOnlySpan<float> weight, int size)
        {
            int vecSize = Vector<float>.Count;
            var ssSim = new Vector<float>(0);
            for (int j = 0; j < size / vecSize; j++)
            {
                var xVec = new Vector<float>(x.Slice(j * vecSize, vecSize));
                ssSim += xVec * xVec;
            }
            float ss = Vector.Sum(ssSim);
            for (int j = (size / vecSize) * vecSize; j < size; j++)
            {
                ss += x[j] * x[j];
            }
            ss /= size;
            ss += 1e-6f;
            ss = 1.0f / (float)Math.Sqrt(ss);
            for (int j = 0; j < size / vecSize; j++)
            {
                var xVec = new Vector<float>(x.Slice(j * vecSize, vecSize));
                var wVec = new Vector<float>(weight.Slice(j * vecSize, vecSize));
                var oVec = (Vector<float>.One + wVec) * (ss * xVec);
                oVec.CopyTo(o.Slice(j * vecSize, vecSize));
            }
            for (int j = (size / vecSize) * vecSize; j < size; j++)
            {
                o[j] = (1.0f + weight[j]) * (ss * x[j]);
            }
        }

        public static void Softmax(Span<float> x)
        {
            if (x.IsEmpty) return;
            float maxVal = x[0];
            for (int i = 1; i < x.Length; i++)
                if (x[i] > maxVal) maxVal = x[i];
            float sum = 0.0f;
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = (float)Math.Exp(x[i] - maxVal);
                sum += x[i];
            }
            if (sum == 0) return;
            for (int i = 0; i < x.Length; i++) x[i] /= sum;
        }

        public static void Matmul(Memory<float> xout, ReadOnlyMemory<float> x, ReadOnlyMemory<float> w)
        {
            int n = x.Length;
            int d = xout.Length;
            
            Parallel.For(0, d, i =>
            {
                var xSpan = x.Span;
                var wRow = w.Span.Slice(i * n, n);
                int vecSize = Vector<float>.Count;

                var sumVec = Vector<float>.Zero;
                for (int j = 0; j < n / vecSize; j++)
                {
                    var xVec = new Vector<float>(xSpan.Slice(j * vecSize, vecSize));
                    var wVec = new Vector<float>(wRow.Slice(j * vecSize, vecSize));
                    sumVec += xVec * wVec;
                }
                float val = Vector.Sum(sumVec);
                for (int j = (n / vecSize) * vecSize; j < n; j++)
                {
                    val += xSpan[j] * wRow[j];
                }
                xout.Span[i] = val;
            });
        }
        
        public static void QMatmul(Memory<float> xout, Quantization.MutableQuantizedTensor x, Quantization.QuantizedTensor w, int n, int gs)
        {
            if (gs == 0) return;

            int d = xout.Length;
            Parallel.For(0, d, i =>
            {
                var xQSpan = x.Q.Span;
                var xSSpan = x.S.Span;
                var wQSpan = w.Q.Span;
                var wSSpan = w.S.Span;

                int ni = i * n;
                float totalSum = 0;
                
                for (int j = 0; j <= n - gs; j += gs)
                {
                    var xGroupSpan = xQSpan.Slice(j, gs);
                    var wGroupSpan = wQSpan.Slice(ni + j, gs);
                    
                    int iVal = 0;
                    int sbyteVecSize = Vector<sbyte>.Count;
                    int k_limit = gs / sbyteVecSize;

                    for (int k = 0; k < k_limit; k++)
                    {
                        var xVecSbyte = new Vector<sbyte>(xGroupSpan.Slice(k * sbyteVecSize, sbyteVecSize));
                        var wVecSbyte = new Vector<sbyte>(wGroupSpan.Slice(k * sbyteVecSize, sbyteVecSize));
                        
                        Vector.Widen(xVecSbyte, out Vector<short> xVecShortL, out Vector<short> xVecShortH);
                        Vector.Widen(wVecSbyte, out Vector<short> wVecShortL, out Vector<short> wVecShortH);
                        
                        iVal += Vector.Dot(xVecShortL, wVecShortL);
                        iVal += Vector.Dot(xVecShortH, wVecShortH);
                    }
                    
                    for (int k = k_limit * sbyteVecSize; k < gs; k++)
                    {
                        iVal += xGroupSpan[k] * wGroupSpan[k];
                    }

                    totalSum += iVal * wSSpan[(ni + j) / gs] * xSSpan[j / gs];
                }
                xout.Span[i] = totalSum;
            });
        }
    }
    
    public class Sampler
    {
        private readonly uint _vocabSize;
        private readonly ProbIndex[] _probIndex;
        private readonly float _temperature;
        private readonly float _topP;
        private ulong _seed;
        private struct ProbIndex { public float Prob; public int Index; }
        private class ProbIndexComparer : IComparer<ProbIndex> { public int Compare(ProbIndex a, ProbIndex b) => b.Prob.CompareTo(a.Prob); }
        private static readonly ProbIndexComparer Comparer = new ProbIndexComparer();
        public Sampler(uint vocabSize, float temperature, float topP, ulong seed) { _vocabSize = vocabSize; _temperature = temperature; _topP = topP; _seed = seed; _probIndex = new ProbIndex[vocabSize]; }
        private static int SampleArgmax(ReadOnlySpan<float> probabilities) { int maxI = 0; float maxP = probabilities[0]; for (int i = 1; i < probabilities.Length; i++) { if (probabilities[i] > maxP) { maxI = i; maxP = probabilities[i]; } } return maxI; }
        private static int SampleMult(ReadOnlySpan<float> probabilities, float rand) { float cdf = 0.0f; for (int i = 0; i < probabilities.Length; i++) { cdf += probabilities[i]; if (rand < cdf) return i; } return probabilities.Length - 1; }
        private int SampleTopP(ReadOnlySpan<float> probabilities, float topP, float rand)
        {
            int n = probabilities.Length; int n0 = 0; float cutoff = (1.0f - topP) / (n - 1);
            for (int i = 0; i < n; i++) { if (probabilities[i] >= cutoff) { _probIndex[n0].Index = i; _probIndex[n0].Prob = probabilities[i]; n0++; } }
            Array.Sort(_probIndex, 0, n0, Comparer); float cumulativeProb = 0.0f; int lastIdx = n0 - 1;
            for (int i = 0; i < n0; i++) { cumulativeProb += _probIndex[i].Prob; if (cumulativeProb > topP) { lastIdx = i; break; } }
            float r = rand * cumulativeProb; float cdf = 0.0f;
            for (int i = 0; i <= lastIdx; i++) { cdf += _probIndex[i].Prob; if (r < cdf) return _probIndex[i].Index; }
            return _probIndex[lastIdx].Index;
        }
        public uint Sample(Span<float> logits)
        {
            int next;
            if (_temperature == 0.0f) { next = SampleArgmax(logits); }
            else
            {
                for (int i = 0; i < logits.Length; i++) logits[i] /= _temperature;
                Functional.Softmax(logits); float rand = Functional.RandomF32(ref _seed);
                if (_topP <= 0.0f || _topP >= 1.0f) { next = SampleMult(logits, rand); }
                else { next = SampleTopP(logits, _topP, rand); }
            }
            return (uint)next;
        }
    }
    
    public class Tokenizer
    {
        private readonly string[] _vocab;
        private readonly float[] _vocabScores;
        private readonly TokenIndex[] _sortedVocab;
        private struct TokenIndex { public string Text; public int Id; }
        public Tokenizer(string path)
        {
            // This part was correct, as tokenizer.bin has a simpler structure.
            var data = File.ReadAllBytes(path); var span = data.AsSpan();
            uint vocabSize = Functional.SliceToU32(span.Slice(0, 4)); _vocab = new string[vocabSize]; _vocabScores = new float[vocabSize];
            int offset = 8;
            for (int i = 0; i < vocabSize; i++)
            {
                _vocabScores[i] = Functional.SliceToF32(span.Slice(offset, 4)); offset += 4;
                int strLen = (int)Functional.SliceToU32(span.Slice(offset, 4)); offset += 4;
                _vocab[i] = Encoding.UTF8.GetString(span.Slice(offset, strLen)); offset += strLen;
            }
            var sortedVocabList = new List<TokenIndex>();
            for (int i = 0; i < vocabSize; i++) sortedVocabList.Add(new TokenIndex { Text = _vocab[i], Id = i });
            sortedVocabList.Sort((a, b) => string.CompareOrdinal(a.Text, b.Text)); _sortedVocab = sortedVocabList.ToArray();
        }
        private int FindToken(string text)
        {
            int low = 0, high = _sortedVocab.Length - 1;
            while (low <= high) { int mid = low + (high - low) / 2; int cmp = string.CompareOrdinal(_sortedVocab[mid].Text, text); if (cmp == 0) return mid; if (cmp < 0) low = mid + 1; else high = mid - 1; }
            return -1;
        }
        public List<uint> Encode(string text, bool bos, bool eos, bool chatFormat)
        {
            var tokens = new List<uint>();
            if (bos) tokens.Add(2); if (chatFormat) tokens.AddRange(new uint[] { 106, 1645, 108 });
            foreach (char c in text) { string cStr = c.ToString(); int id = FindToken(cStr); if (id != -1) { tokens.Add((uint)_sortedVocab[id].Id); } else { foreach (byte b in Encoding.UTF8.GetBytes(cStr)) { tokens.Add((uint)b + 3); } } }
            while (true)
            {
                float bestScore = -1e10f; uint bestId = 0; int bestIdx = -1;
                for (int i = 0; i < tokens.Count - 1; i++) { string newT = _vocab[tokens[i]] + _vocab[tokens[i + 1]]; int id = FindToken(newT); if (id != -1) { var tempT = _sortedVocab[id]; if (_vocabScores[tempT.Id] > bestScore) { bestScore = _vocabScores[tempT.Id]; bestId = (uint)tempT.Id; bestIdx = i; } } }
                if (bestIdx == -1) break;
                tokens[bestIdx] = bestId; tokens.RemoveAt(bestIdx + 1);
            }
            if (chatFormat) tokens.AddRange(new uint[] { 107, 108, 106, 2516, 108 }); if (eos) tokens.Add(1);
            return tokens;
        }
        public string Decode(uint token)
        {
            string piece = _vocab[token];
            if (piece.StartsWith("<0x") && piece.EndsWith('>') && piece.Length == 6) { if (byte.TryParse(piece.AsSpan(3, 2), System.Globalization.NumberStyles.HexNumber, null, out byte byteVal)) return ((char)byteVal).ToString(); }
            return piece;
        }
    }
    
    public class Transformer
    {
        // ИСПРАВЛЕНИЕ: Структура теперь точно соответствует первым 33 байтам заголовка из Python.
        // GroupSize убран, а QType теперь имеет тип byte.
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct TransformerArgs
        {
            public uint Dim;
            public uint HiddenDim;
            public uint NLayers;
            public uint NHeads;
            public uint HeadSize;
            public uint NKvHeads;
            public uint VocabSize;
            public uint SeqLen;
            public Quantization.QuantType QType;
        }

        private class TransformerWeights
        {
            public ReadOnlyMemory<float> TokenEmbeddingTable = ReadOnlyMemory<float>.Empty;
            public ReadOnlyMemory<float>[]? Wq, Wk, Wv, Wo;
            public Quantization.QuantizedTensor[]? WqQuant, WkQuant, WvQuant, WoQuant;
            public ReadOnlyMemory<float>[] WRmsAtt = Array.Empty<ReadOnlyMemory<float>>();
            public ReadOnlyMemory<float>[]? W1, W2, W3;
            public Quantization.QuantizedTensor[]? W1Quant, W2Quant, W3Quant;
            public ReadOnlyMemory<float>[] WRmsPostAtt = Array.Empty<ReadOnlyMemory<float>>();
            public ReadOnlyMemory<float>[] WRmsPreFfn = Array.Empty<ReadOnlyMemory<float>>();
            public ReadOnlyMemory<float>[] WRmsPostFfn = Array.Empty<ReadOnlyMemory<float>>();
            public ReadOnlyMemory<float> WRmsFinal = ReadOnlyMemory<float>.Empty;
            public ReadOnlyMemory<float>? WCls;
            public Quantization.QuantizedTensor? WClsQuant;
        }

        private class TransformerState
        {
            public float[] X = Array.Empty<float>();
            public float[] Xb = Array.Empty<float>();
            public float[] Xb2 = Array.Empty<float>();
            public float[] Hb = Array.Empty<float>();
            public float[] Hb2 = Array.Empty<float>();
            public float[] Q = Array.Empty<float>();
            public float[] Logits = Array.Empty<float>();
            public Quantization.MutableQuantizedTensor? Xq, Hq;
            public float[] KeyCache = Array.Empty<float>();
            public float[] ValueCache = Array.Empty<float>();
        }

        public readonly TransformerArgs Args;
        private readonly TransformerWeights _weights;
        private readonly TransformerState _state;
        // ИСПРАВЛЕНИЕ: GroupSize теперь хранится как отдельное поле класса.
        private readonly uint _groupSize;

        public unsafe Transformer(string modelPath)
        {
            using var file = File.OpenRead(modelPath);
            using var mmf = MemoryMappedFile.CreateFromFile(file, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, false);
            using var view = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            
            byte* basePtr = (byte*)0;
            view.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
            
            if (basePtr == null) throw new InvalidOperationException("Failed to acquire memory map pointer.");

            if (Functional.SliceToU32(new ReadOnlySpan<byte>(basePtr, 4)) != 0x73726d6c)
                 throw new InvalidDataException("Model not in llm.rs format.");

            Console.WriteLine($"LMRS version: {Functional.SliceToU32(new ReadOnlySpan<byte>(basePtr + 4, 4))}");

            // ИСПРАВЛЕНИЕ: Читаем заголовок по частям, в точности как его пишет Python.
            long currentOffset = 8;
            Args = MemoryMarshal.Read<TransformerArgs>(new ReadOnlySpan<byte>(basePtr + currentOffset, sizeof(TransformerArgs)));
            currentOffset += sizeof(TransformerArgs);

            _groupSize = BinaryPrimitives.ReadUInt32LittleEndian(new ReadOnlySpan<byte>(basePtr + currentOffset, 4));
            
            _weights = new TransformerWeights();
            long weightsOffset = 48; // Данные весов начинаются с фиксированного смещения 48

            bool quantized = Args.QType != Quantization.QuantType.None;
            if (quantized) Console.WriteLine("Using Q8_0 quantization.\n");

            uint kvDim = Args.HeadSize * Args.NKvHeads;
            
            if (!quantized)
            {
                _weights.TokenEmbeddingTable = GetArray<float>(basePtr, ref weightsOffset, Args.VocabSize * Args.Dim);
                _weights.WRmsAtt = GetArrays<float>(basePtr, ref weightsOffset, Args.NLayers, Args.Dim);
                _weights.Wq = GetArrays<float>(basePtr, ref weightsOffset, Args.NLayers, Args.Dim * Args.NHeads * Args.HeadSize);
                _weights.Wk = GetArrays<float>(basePtr, ref weightsOffset, Args.NLayers, Args.Dim * kvDim);
                _weights.Wv = GetArrays<float>(basePtr, ref weightsOffset, Args.NLayers, Args.Dim * kvDim);
                _weights.Wo = GetArrays<float>(basePtr, ref weightsOffset, Args.NLayers, Args.Dim * Args.NHeads * Args.HeadSize);
                _weights.WRmsPostAtt = GetArrays<float>(basePtr, ref weightsOffset, Args.NLayers, Args.Dim);
                _weights.WRmsPreFfn = GetArrays<float>(basePtr, ref weightsOffset, Args.NLayers, Args.Dim);
                _weights.W1 = GetArrays<float>(basePtr, ref weightsOffset, Args.NLayers, Args.Dim * Args.HiddenDim);
                _weights.W2 = GetArrays<float>(basePtr, ref weightsOffset, Args.NLayers, Args.Dim * Args.HiddenDim);
                _weights.W3 = GetArrays<float>(basePtr, ref weightsOffset, Args.NLayers, Args.Dim * Args.HiddenDim);
                _weights.WRmsPostFfn = GetArrays<float>(basePtr, ref weightsOffset, Args.NLayers, Args.Dim);
                _weights.WRmsFinal = GetArray<float>(basePtr, ref weightsOffset, Args.Dim);
                _weights.WCls = _weights.TokenEmbeddingTable;
            }
            else
            {
                var embTabQuant = GetQuantizedTensor(basePtr, ref weightsOffset, Args.VocabSize * Args.Dim, _groupSize);
                var embTab = new float[Args.VocabSize * Args.Dim];
                Quantization.Dequantize(embTabQuant, embTab, embTab.Length, _groupSize);
                _weights.TokenEmbeddingTable = embTab;

                _weights.WRmsAtt = GetArrays<float>(basePtr, ref weightsOffset, Args.NLayers, Args.Dim);
                _weights.WqQuant = GetQuantizedTensors(basePtr, ref weightsOffset, Args.NLayers, Args.Dim * Args.NHeads * Args.HeadSize, _groupSize);
                _weights.WkQuant = GetQuantizedTensors(basePtr, ref weightsOffset, Args.NLayers, Args.Dim * kvDim, _groupSize);
                _weights.WvQuant = GetQuantizedTensors(basePtr, ref weightsOffset, Args.NLayers, Args.Dim * kvDim, _groupSize);
                _weights.WoQuant = GetQuantizedTensors(basePtr, ref weightsOffset, Args.NLayers, Args.Dim * Args.NHeads * Args.HeadSize, _groupSize);
                _weights.WRmsPostAtt = GetArrays<float>(basePtr, ref weightsOffset, Args.NLayers, Args.Dim);
                _weights.WRmsPreFfn = GetArrays<float>(basePtr, ref weightsOffset, Args.NLayers, Args.Dim);
                _weights.W1Quant = GetQuantizedTensors(basePtr, ref weightsOffset, Args.NLayers, Args.Dim * Args.HiddenDim, _groupSize);
                _weights.W2Quant = GetQuantizedTensors(basePtr, ref weightsOffset, Args.NLayers, Args.Dim * Args.HiddenDim, _groupSize);
                _weights.W3Quant = GetQuantizedTensors(basePtr, ref weightsOffset, Args.NLayers, Args.Dim * Args.HiddenDim, _groupSize);
                _weights.WRmsPostFfn = GetArrays<float>(basePtr, ref weightsOffset, Args.NLayers, Args.Dim);
                _weights.WRmsFinal = GetArray<float>(basePtr, ref weightsOffset, Args.Dim);
                _weights.WClsQuant = embTabQuant;
            }

            _state = new TransformerState
            {
                X = new float[Args.Dim],
                Xb = new float[Args.Dim],
                Xb2 = new float[Args.Dim],
                Hb = new float[Args.HiddenDim],
                Hb2 = new float[Args.HiddenDim],
                Q = new float[Args.NHeads * Args.HeadSize],
                KeyCache = new float[Args.NLayers * Args.SeqLen * kvDim],
                ValueCache = new float[Args.NLayers * Args.SeqLen * kvDim],
                Logits = new float[Args.VocabSize]
            };
            
            if (quantized && _groupSize > 0)
            {
                _state.Xq = new Quantization.MutableQuantizedTensor(new sbyte[Args.Dim], new float[Args.Dim / _groupSize]);
                _state.Hq = new Quantization.MutableQuantizedTensor(new sbyte[Args.HiddenDim], new float[Args.HiddenDim / _groupSize]);
            }
             
            view.SafeMemoryMappedViewHandle.ReleasePointer();
        }

        private unsafe ReadOnlyMemory<T> GetArray<T>(byte* basePtr, ref long offset, uint count) where T : struct
        {
            int size = (int)(count * Unsafe.SizeOf<T>());
            var array = new T[count];
            new ReadOnlySpan<byte>(basePtr + offset, size).CopyTo(MemoryMarshal.AsBytes(array.AsSpan()));
            offset += size;
            return array;
        }

        private unsafe ReadOnlyMemory<T>[] GetArrays<T>(byte* basePtr, ref long offset, uint n, uint sizeEach) where T : struct
        {
            var arrays = new ReadOnlyMemory<T>[n];
            for(int i = 0; i < n; i++)
            {
                arrays[i] = GetArray<T>(basePtr, ref offset, sizeEach);
            }
            return arrays;
        }
        
        private unsafe Quantization.QuantizedTensor GetQuantizedTensor(byte* basePtr, ref long offset, uint sizeEach, uint gs)
        {
            var q = GetArray<sbyte>(basePtr, ref offset, sizeEach);
            var s = (gs > 0) ? GetArray<float>(basePtr, ref offset, sizeEach / gs) : Array.Empty<float>();
            return new Quantization.QuantizedTensor(q, s);
        }

        private unsafe Quantization.QuantizedTensor[] GetQuantizedTensors(byte* basePtr, ref long offset, uint n, uint sizeEach, uint gs)
        {
            var tensors = new Quantization.QuantizedTensor[n];
            for (int i = 0; i < n; i++)
            {
                tensors[i] = GetQuantizedTensor(basePtr, ref offset, sizeEach, gs);
            }
            return tensors;
        }

        public Span<float> Forward(uint token, uint pos)
        {
            var p = Args;
            var w = _weights;
            var s = _state;
            var x = s.X.AsSpan();
            
            uint dim = p.Dim;
            uint headSize = p.HeadSize;
            uint attDim = p.NHeads * headSize;
            uint kvDim = p.HeadSize * p.NKvHeads;
            uint kvMul = p.NHeads / p.NKvHeads;
            uint hiddenDim = p.HiddenDim;
            uint gs = _groupSize; // Используем _groupSize вместо Args.GroupSize
            bool quantized = p.QType != Quantization.QuantType.None;

            w.TokenEmbeddingTable.Span.Slice((int)(token * dim), (int)dim).CopyTo(x);

            float normalizer = (float)Math.Sqrt(dim);
            for (int i = 0; i < x.Length; i++) x[i] *= normalizer;

            for (uint l = 0; l < p.NLayers; l++)
            {
                Functional.RmsNorm(s.Xb, x, w.WRmsAtt[l].Span, (int)dim);

                int loff = (int)(l * p.SeqLen * kvDim);
                var k = s.KeyCache.AsMemory().Slice((int)(loff + pos * kvDim), (int)kvDim);
                var v = s.ValueCache.AsMemory().Slice((int)(loff + pos * kvDim), (int)kvDim);
                
                var qMem = s.Q.AsMemory();
                var xbMem = s.Xb.AsMemory();

                if (!quantized)
                {
                    Functional.Matmul(qMem, xbMem, w.Wq![l]);
                    Functional.Matmul(k, xbMem, w.Wk![l]);
                    Functional.Matmul(v, xbMem, w.Wv![l]);
                }
                else
                {
                    var sxq = s.Xq!;
                    Quantization.Quantize(sxq, xbMem.Span, (int)dim, gs);
                    Functional.QMatmul(qMem, sxq, w.WqQuant![l], (int)dim, (int)gs);
                    Functional.QMatmul(k, sxq, w.WkQuant![l], (int)dim, (int)gs);
                    Functional.QMatmul(v, sxq, w.WvQuant![l], (int)dim, (int)gs);
                }
                
                // RoPE from original Rust code
                for (uint i = 0; i < p.NHeads; i++)
                {
                    for (uint j = 0; j < headSize / 2; j++)
                    {
                        uint headDim = j * 2;
                        float freq = 1.0f / (float)Math.Pow(10000.0f, headDim / (float)headSize);
                        float val = pos * freq;
                        float fcr = (float)Math.Cos(val);
                        float fci = (float)Math.Sin(val);
                        
                        var q_slice = s.Q.AsSpan();
                        float q0 = q_slice[(int)(i * headSize + j)];
                        float q1 = q_slice[(int)(i * headSize + j + headSize / 2)];
                        q_slice[(int)(i * headSize + j)] = q0 * fcr - q1 * fci;
                        q_slice[(int)(i * headSize + j + headSize/2)] = q0 * fci + q1 * fcr;
                        
                        if ((i * headSize) + j + headSize/2 < kvDim) 
                        {
                            var k_slice = k.Span;
                            float k0 = k_slice[(int)(i * headSize + j)];
                            float k1 = k_slice[(int)(i * headSize + j + headSize / 2)];
                            k_slice[(int)(i * headSize + j)] = k0 * fcr - k1 * fci;
                            k_slice[(int)(i * headSize + j + headSize/2)] = k0 * fci + k1 * fcr;
                        }
                    }
                }
                
                var headAttMem = s.Xb.AsMemory(0, (int)attDim);
                Parallel.For(0, (int)p.NHeads, h =>
                {
                    var q = s.Q.AsSpan().Slice(h * (int)headSize, (int)headSize);
                    var xb_h = headAttMem.Span.Slice(h * (int)headSize, (int)headSize); 
                    var att = new float[pos + 1];

                    for (uint t = 0; t <= pos; t++)
                    {
                        var k_slice = s.KeyCache.AsSpan().Slice((int)(loff + t * kvDim + (h/kvMul) * headSize), (int)headSize);
                        float score = 0;
                        for(int i = 0; i < headSize; i++) score += q[i] * k_slice[i];
                        
                        score /= (float)Math.Sqrt(256.0f);
                        score /= 50.0f;
                        score = (float)Math.Tanh(score);
                        score *= 50.0f;
                        score += (pos - t <= 4096) ? 0.0f : -2.3819763e38f;
                        att[t] = score;
                    }
                    Functional.Softmax(att);
                    
                    xb_h.Fill(0.0f);
                    for (uint t = 0; t <= pos; t++)
                    {
                        var v_slice = s.ValueCache.AsSpan().Slice((int)(loff + t * kvDim + (h/kvMul) * headSize), (int)headSize);
                        float a = att[t];
                        for(int i = 0; i < headSize; i++) xb_h[i] += a * v_slice[i];
                    }
                });
                
                var xb2Mem = s.Xb2.AsMemory();
                if (!quantized)
                {
                    Functional.Matmul(xb2Mem, headAttMem, w.Wo![l]);
                }
                else
                {
                    var sxq = s.Xq!;
                    Quantization.Quantize(sxq, headAttMem.Span, (int)attDim, gs);
                    Functional.QMatmul(xb2Mem, sxq, w.WoQuant![l], (int)attDim, (int)gs);
                }

                Functional.RmsNorm(s.Xb, s.Xb2.AsSpan(), w.WRmsPostAtt[l].Span, (int)dim);
                for (int i = 0; i < dim; i++) x[i] += s.Xb[i];
                
                Functional.RmsNorm(s.Xb, x, w.WRmsPreFfn[l].Span, (int)dim);
                
                var hbMem = s.Hb.AsMemory();
                var hb2Mem = s.Hb2.AsMemory();
                var xbMemFfn = s.Xb.AsMemory();
                if (!quantized)
                {
                    Functional.Matmul(hbMem, xbMemFfn, w.W1![l]);
                    Functional.Matmul(hb2Mem, xbMemFfn, w.W3![l]);
                }
                else
                {
                    var sxq = s.Xq!;
                    Quantization.Quantize(sxq, xbMemFfn.Span, (int)dim, gs);
                    Functional.QMatmul(hbMem, sxq, w.W1Quant![l], (int)dim, (int)gs);
                    Functional.QMatmul(hb2Mem, sxq, w.W3Quant![l], (int)dim, (int)gs);
                }
                
                for(int i = 0; i < hiddenDim; i++)
                {
                    float val = s.Hb[i];
                    val *= 0.5f * (1.0f + (float)Math.Tanh(0.7978845608028654f * (val + 0.044715f * val * val * val)));
                    val *= s.Hb2[i];
                    s.Hb[i] = val;
                }

                if (!quantized)
                {
                     Functional.Matmul(xbMemFfn, hbMem, w.W2![l]);
                }
                else
                {
                    var shq = s.Hq!;
                    Quantization.Quantize(shq, hbMem.Span, (int)hiddenDim, gs);
                    Functional.QMatmul(xbMemFfn, shq, w.W2Quant![l], (int)hiddenDim, (int)gs);
                }

                Functional.RmsNorm(s.Xb2, s.Xb, w.WRmsPostFfn[l].Span, (int)dim);
                for (int i = 0; i < dim; i++) x[i] += s.Xb2[i];
            }

            x.CopyTo(s.Xb.AsSpan());
            
            Functional.RmsNorm(x, s.Xb, w.WRmsFinal.Span, (int)dim);

            var logitsMem = s.Logits.AsMemory();
            var xMem = s.X.AsMemory();
            if (!quantized)
            {
                Functional.Matmul(logitsMem, xMem, w.WCls!.Value);
            }
            else
            {
                var sxq = s.Xq!;
                Quantization.Quantize(sxq, x, (int)dim, gs);
                Functional.QMatmul(logitsMem, sxq, w.WClsQuant!, (int)dim, (int)gs);
            }

            for (int d = 0; d < p.Dim; d++)
            {
                s.Logits[d] /= 30.0f;
                s.Logits[d] = (float)Math.Tanh(s.Logits[d]);
                s.Logits[d] *= 30.0f;
            }

            return s.Logits;
        }
    }
    
    public class Program
    {
        public static void Main(string[] args)
        {
            var modelOption = new Option<string>("--model", "Path to the model file.") { IsRequired = true };
            var tokenizerOption = new Option<string>("--tokenizer", () => "tokenizer.bin", "Path to the tokenizer file.");
            var tempOption = new Option<float>("--temperature", () => 1.0f, "Temperature for sampling.");
            var topPOption = new Option<float>("--top-p", () => 0.9f, "Top-p for sampling.");
            var seedOption = new Option<ulong?>("--seed", "Random seed.");
            var metricsOption = new Option<bool>("--show-metrics", () => false, "Show performance metrics.");

            var rootCommand = new RootCommand("Gemma-2B inference in C#");
            rootCommand.AddOption(modelOption);
            rootCommand.AddOption(tokenizerOption);
            rootCommand.AddOption(tempOption);
            rootCommand.AddOption(topPOption);
            rootCommand.AddOption(seedOption);
            rootCommand.AddOption(metricsOption);

            rootCommand.SetHandler(context =>
            {
                var modelPath = context.ParseResult.GetValueForOption(modelOption)!;
                var tokenizerPath = context.ParseResult.GetValueForOption(tokenizerOption)!;
                var temperature = context.ParseResult.GetValueForOption(tempOption);
                var topP = context.ParseResult.GetValueForOption(topPOption);
                var seedValue = context.ParseResult.GetValueForOption(seedOption);
                var showMetrics = context.ParseResult.GetValueForOption(metricsOption);

                RunChat(modelPath, tokenizerPath, temperature, topP, seedValue, showMetrics);
            });
            
            rootCommand.Invoke(args);
        }

        static void RunChat(string modelPath, string tokenizerPath, float temperature, float topP, ulong? seedValue, bool showMetrics)
        {
            string logo = @"
    L      M     M  RRRR    ssss
    L      MM   MM  R   R  s
    L      M M M M  RRRR    sss
    L      M  M  M  R  R       s
    LLLL   M     M  R   R  sssss
    ";
            Console.WriteLine(logo);

            if (!File.Exists(tokenizerPath)) { Console.Error.WriteLine($"Tokenizer file not found: {tokenizerPath}"); return; }
            if (!File.Exists(modelPath)) { Console.Error.WriteLine($"Model file not found: {modelPath}"); return; }
            
            var tokenizer = new Tokenizer(tokenizerPath);
            var model = new Transformer(modelPath);

            ulong seed = seedValue ?? (ulong)DateTime.UtcNow.Ticks;
            var sampler = new Sampler(model.Args.VocabSize, temperature, topP, seed);

            bool userTurn = true;
            int userIdx = 0;
            uint pos = 0;
            uint token = 0, next = 0;
            int numPromptTokens = 0;
            double totalGenTokens = 0;
            double totalDurationMs = 0;

            List<uint> promptTokens = new List<uint>();

            Console.OutputEncoding = Encoding.UTF8;

            while (true)
            {
                if (userTurn)
                {
                    Console.Write("You: ");
                    string userPrompt = Console.ReadLine() ?? "";
                    
                    promptTokens = tokenizer.Encode(userPrompt.Trim(), true, false, true);
                    numPromptTokens = promptTokens.Count;
                    userIdx = 0;
                    
                    Console.Write("Assistant: ");
                    Console.Out.Flush();
                    
                    userTurn = false;
                }
                
                if (pos >= model.Args.SeqLen)
                {
                    pos = 0;
                    userTurn = true;
                    Console.WriteLine("\n[Context length reached]");
                    continue;
                }

                if (userIdx < numPromptTokens)
                {
                    token = promptTokens[userIdx];
                    userIdx++;
                }
                else
                {
                    token = next;
                }

                if (token == 1) // <eos> token
                {
                    userTurn = true;
                    Console.WriteLine();
                    if (showMetrics && totalDurationMs > 0)
                    {
                        Console.WriteLine($"\nSpeed: {totalGenTokens / (totalDurationMs / 1000.0):F2} tok/s");
                        totalDurationMs = 0;
                        totalGenTokens = 0;
                    }
                    continue;
                }
                
                var stopwatch = Stopwatch.StartNew();
                var logits = model.Forward(token, pos);
                next = sampler.Sample(logits);
                stopwatch.Stop();
                
                pos++;

                if (userIdx >= numPromptTokens && next != 1)
                {
                    string piece = tokenizer.Decode(token);
                    Console.Write(piece);
                    Console.Out.Flush();
                }   
                
                if(userIdx > numPromptTokens)
                {
                    totalDurationMs += stopwatch.Elapsed.TotalMilliseconds;
                    totalGenTokens++;
                }
            }
        }
    }
}
// --- КОНЕЦ ФИНАЛЬНОГО ОБЪЕДИНЕННОГО ФАЙЛА C# ---