using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Niyah.App;

public static class NiyahBridge
{
    private const string DllName = "niyah";

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr niyah_bridge_version();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int niyah_bridge_add_document(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string content,
        out IntPtr docIdOut);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr niyah_bridge_get_document(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string docId);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int niyah_bridge_delete_document(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string docId);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr niyah_bridge_search_json(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string query,
        int maxHits);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int niyah_bridge_document_count();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void niyah_bridge_free_string(IntPtr text);

    public static string Version =>
        Marshal.PtrToStringUTF8(niyah_bridge_version()) ?? "unknown";

    public static int DocumentCount => niyah_bridge_document_count();

    public static string AddDocument(string content)
    {
        int status = niyah_bridge_add_document(content, out IntPtr idPointer);
        try
        {
            if (status != 0 || idPointer == IntPtr.Zero)
            {
                throw new InvalidOperationException($"Native add-document failed with status {status}.");
            }

            return Marshal.PtrToStringUTF8(idPointer)
                ?? throw new InvalidOperationException("Native add-document returned an invalid UTF-8 id.");
        }
        finally
        {
            if (idPointer != IntPtr.Zero)
            {
                niyah_bridge_free_string(idPointer);
            }
        }
    }

    public static string? GetDocument(string docId)
    {
        IntPtr pointer = niyah_bridge_get_document(docId);
        if (pointer == IntPtr.Zero)
        {
            return null;
        }

        try
        {
            return Marshal.PtrToStringUTF8(pointer);
        }
        finally
        {
            niyah_bridge_free_string(pointer);
        }
    }

    public static bool DeleteDocument(string docId) =>
        niyah_bridge_delete_document(docId) == 0;

    public static List<SearchResult> Search(string query, int maxHits = 100)
    {
        if (maxHits <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxHits));
        }

        IntPtr pointer = niyah_bridge_search_json(query, maxHits);
        if (pointer == IntPtr.Zero)
        {
            throw new InvalidOperationException("Native search failed to return a result buffer.");
        }

        try
        {
            string json = Marshal.PtrToStringUTF8(pointer)
                ?? throw new InvalidOperationException("Native search returned invalid UTF-8 JSON.");
            var nativeResults = JsonSerializer.Deserialize<List<BridgeSearchResult>>(json)
                ?? new List<BridgeSearchResult>();

            var results = new List<SearchResult>(nativeResults.Count);
            foreach (BridgeSearchResult item in nativeResults)
            {
                results.Add(new SearchResult(item.Id, item.Snippet, item.Score));
            }
            return results;
        }
        finally
        {
            niyah_bridge_free_string(pointer);
        }
    }

    private sealed class BridgeSearchResult
    {
        [JsonPropertyName("id")]
        public string Id { get; set; } = string.Empty;

        [JsonPropertyName("snippet")]
        public string Snippet { get; set; } = string.Empty;

        [JsonPropertyName("score")]
        public float Score { get; set; }
    }
}

public sealed record SearchResult(string DocId, string Snippet, float Score);
