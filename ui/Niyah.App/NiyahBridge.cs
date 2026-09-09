using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Win32.SafeHandles;

namespace Niyah.App;

internal static class NiyahNative
{
    internal const string DllName = "niyah";

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr niyah_bridge_version();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int niyah_bridge_open(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string conninfo,
        out IntPtr context);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void niyah_bridge_close(IntPtr context);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr niyah_bridge_search_json(
        NiyahBridgeHandle context,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string query,
        int maxHits);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void niyah_bridge_free_string(IntPtr text);
}

internal sealed class NiyahBridgeHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    internal NiyahBridgeHandle(IntPtr handle) : base(ownsHandle: true)
    {
        SetHandle(handle);
    }

    protected override bool ReleaseHandle()
    {
        NiyahNative.niyah_bridge_close(handle);
        return true;
    }
}

public sealed class NiyahBridge : IDisposable
{
    private readonly NiyahBridgeHandle _handle;
    private bool _disposed;

    public NiyahBridge(string connectionString)
    {
        if (string.IsNullOrWhiteSpace(connectionString))
            throw new ArgumentException("PostgreSQL connection string is required.", nameof(connectionString));

        int rc = NiyahNative.niyah_bridge_open(connectionString, out IntPtr raw);
        if (rc != 0 || raw == IntPtr.Zero)
            throw new InvalidOperationException($"niyah_bridge_open failed: {rc}");

        _handle = new NiyahBridgeHandle(raw);
    }

    public static string Version =>
        Marshal.PtrToStringUTF8(NiyahNative.niyah_bridge_version()) ?? "unknown";

    public IReadOnlyList<SearchResult> Search(string query, int maxHits = 20)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (string.IsNullOrWhiteSpace(query))
            return Array.Empty<SearchResult>();
        if (maxHits is < 1 or > 200)
            throw new ArgumentOutOfRangeException(nameof(maxHits));

        IntPtr raw = NiyahNative.niyah_bridge_search_json(_handle, query, maxHits);
        if (raw == IntPtr.Zero)
            throw new InvalidOperationException("Native PostgreSQL search failed.");

        try
        {
            string json = Marshal.PtrToStringUTF8(raw)
                ?? throw new InvalidOperationException("Native bridge returned invalid UTF-8.");

            List<NativeRow>? rows = JsonSerializer.Deserialize<List<NativeRow>>(json);
            if (rows is null || rows.Count == 0)
                return Array.Empty<SearchResult>();

            var result = new List<SearchResult>(rows.Count);
            foreach (NativeRow row in rows)
            {
                result.Add(new SearchResult(
                    row.DocumentId, row.ChunkId, row.Heading, row.Snippet, row.Score));
            }
            return result;
        }
        finally
        {
            NiyahNative.niyah_bridge_free_string(raw);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _handle.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private sealed class NativeRow
    {
        [JsonPropertyName("chunk_id")]
        public string ChunkId { get; init; } = string.Empty;

        [JsonPropertyName("document_id")]
        public string DocumentId { get; init; } = string.Empty;

        [JsonPropertyName("heading")]
        public string? Heading { get; init; }

        [JsonPropertyName("snippet")]
        public string Snippet { get; init; } = string.Empty;

        [JsonPropertyName("score")]
        public float Score { get; init; }
    }
}

public sealed record SearchResult(
    string DocId,
    string ChunkId,
    string? Heading,
    string Snippet,
    float Score);
