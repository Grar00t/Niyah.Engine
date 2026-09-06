using System;
using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Input;

namespace Niyah.App;

public partial class MainWindow : Window
{
    private readonly ObservableCollection<SearchResult> _searchResults = new();
    private readonly ObservableCollection<string> _docIds = new();

    public MainWindow()
    {
        InitializeComponent();
        ResultsListBox.ItemsSource = _searchResults;
        DocListBox.ItemsSource = _docIds;

        try
        {
            VersionLabel.Text = $"v{NiyahBridge.Version}";
            RefreshDocumentCount();
            StatusLabel.Text = "Ready.";
        }
        catch (Exception ex)
        {
            StatusLabel.Text = $"Bridge load error: {ex.Message}";
        }
    }

    private void SearchTextBox_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter)
        {
            RunSearch();
        }
    }

    private void SearchButton_Click(object sender, RoutedEventArgs e) => RunSearch();

    private void RunSearch()
    {
        string query = SearchTextBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(query))
        {
            return;
        }

        try
        {
            List<SearchResult> results = NiyahBridge.Search(query);
            _searchResults.Clear();
            foreach (SearchResult result in results)
            {
                _searchResults.Add(result);
            }

            StatusLabel.Text = results.Count > 0
                ? $"Found {results.Count} result(s) for '{query}'"
                : $"No results for '{query}'";
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
        if (ResultsListBox.SelectedItem is not SearchResult result)
        {
            return;
        }

        try
        {
            SelectedDocLabel.Text = $"Document: {result.DocId}";
            DocPreviewBox.Text = NiyahBridge.GetDocument(result.DocId)
                ?? "(content unavailable)";
        }
        catch (Exception ex)
        {
            StatusLabel.Text = $"Preview error: {ex.Message}";
        }
    }

    private void AddDocumentButton_Click(object sender, RoutedEventArgs e)
    {
        string content = DocumentTextBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(content))
        {
            return;
        }

        try
        {
            string id = NiyahBridge.AddDocument(content);
            _docIds.Add(id);
            DocumentTextBox.Clear();
            AddStatusLabel.Text = $"Added: {id}";
            RefreshDocumentCount();
            StatusLabel.Text = $"Document {id} added.";
        }
        catch (Exception ex)
        {
            AddStatusLabel.Text = "Add failed.";
            StatusLabel.Text = $"Add error: {ex.Message}";
        }
    }

    private void DocListBox_SelectionChanged(
        object sender,
        System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (DocListBox.SelectedItem is not string docId)
        {
            return;
        }

        try
        {
            string? content = NiyahBridge.GetDocument(docId);
            StatusLabel.Text = content is null
                ? $"Document {docId} not found."
                : $"Document {docId} — {content.Length} chars";
        }
        catch (Exception ex)
        {
            StatusLabel.Text = $"Document read error: {ex.Message}";
        }
    }

    private void DeleteDocButton_Click(object sender, RoutedEventArgs e)
    {
        if (DocListBox.SelectedItem is not string docId)
        {
            return;
        }

        try
        {
            if (!NiyahBridge.DeleteDocument(docId))
            {
                StatusLabel.Text = $"Could not delete {docId}.";
                return;
            }

            _docIds.Remove(docId);
            _searchResults.Clear();
            SelectedDocLabel.Text = "Select a result to preview content";
            DocPreviewBox.Clear();
            RefreshDocumentCount();
            StatusLabel.Text = $"Deleted {docId}.";
        }
        catch (Exception ex)
        {
            StatusLabel.Text = $"Delete error: {ex.Message}";
        }
    }

    private void RefreshDocumentCount()
    {
        DocCountLabel.Text = $"{NiyahBridge.DocumentCount} documents";
    }
}
