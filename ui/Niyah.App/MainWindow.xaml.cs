using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Input;

namespace Niyah.App;

public partial class MainWindow : Window
{
    private readonly ObservableCollection<SearchResult> _searchResults = new();
    private readonly ObservableCollection<string> _docIds = new();
    private NiyahBridge? _bridge;

    public MainWindow()
    {
        InitializeComponent();
        ResultsListBox.ItemsSource = _searchResults;
        DocListBox.ItemsSource = _docIds;
        DocCountLabel.Text = "PostgreSQL canonical store";

        try
        {
            string conninfo =
                Environment.GetEnvironmentVariable("NIYAH_PG_CONNINFO") ?? "dbname=niyah";

            _bridge = new NiyahBridge(conninfo);
            VersionLabel.Text = $"v{NiyahBridge.Version}";
            StatusLabel.Text = "PostgreSQL bridge ready.";
        }
        catch (Exception ex)
        {
            StatusLabel.Text = $"Bridge load error: {ex.Message}";
        }

        Closed += (_, _) => _bridge?.Dispose();
    }

    private void SearchTextBox_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter) RunSearch();
    }

    private void SearchButton_Click(object sender, RoutedEventArgs e) => RunSearch();

    private void RunSearch()
    {
        string query = SearchTextBox.Text.Trim();
        if (query.Length == 0 || _bridge is null) return;

        try
        {
            IReadOnlyList<SearchResult> rows = _bridge.Search(query, 20);
            _searchResults.Clear();
            foreach (SearchResult row in rows) _searchResults.Add(row);
            StatusLabel.Text = $"Found {rows.Count} result(s).";
        }
        catch (Exception ex)
        {
            StatusLabel.Text = $"Search error: {ex.Message}";
        }
    }

    private void ResultsListBox_SelectionChanged(
        object sender,
        System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (ResultsListBox.SelectedItem is not SearchResult result) return;

        SelectedDocLabel.Text = $"Document: {result.DocId} / Chunk: {result.ChunkId}";
        DocPreviewBox.Text = result.Snippet;
    }

    private void AddDocumentButton_Click(object sender, RoutedEventArgs e)
    {
        AddStatusLabel.Text = "Disabled: canonical PostgreSQL ingestion is not implemented yet.";
        StatusLabel.Text = "No RAM fallback was used.";
    }

    private void DocListBox_SelectionChanged(
        object sender,
        System.Windows.Controls.SelectionChangedEventArgs e)
    {
        StatusLabel.Text = "Document mutation view disabled until canonical ingestion is implemented.";
    }

    private void DeleteDocButton_Click(object sender, RoutedEventArgs e)
    {
        StatusLabel.Text = "Delete disabled: no canonical PostgreSQL delete contract is implemented.";
    }
}
